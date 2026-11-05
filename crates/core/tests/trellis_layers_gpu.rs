//! GitHub #339 (spec 27b-16gb/05a): the trellis body's layer programs, end to
//! end through the step ABI's layer entry points, against the f64 layer
//! oracle (P1-20, `f64_reference::evaluate_layer`) evaluated over the same
//! trellis weights.
//!
//! The oracle reads the served artifact's plane names and orders, so this
//! builds, per layer checked, a small artifact that holds exactly those: each
//! of the layer's trellis matrices decoded (`ignis_trellis_reconstruct`, bit
//! for bit exllamav3's inner matrix) and rotated back on the host,
//! `W = diag(suh) H inner H diag(svh)` (the op's header), and laid out in the
//! served plane's rows -- `gdn/in_proj_qkv` then `gdn/in_proj_z` as
//! `gdn/query_key_value_z`, `attention/q_proj`'s per-head [q | gate] halves
//! split into the q and gate blocks of `attention/query_key_gate_value` around
//! `k_proj`, with `v_proj` last, `mlp/gate_proj` then `up_proj` as
//! `mlp/gate_up` -- in BF16; the layer's other tensors are its own bytes. That
//! layout is written here, independently of the leaf's: the q/gate split, the
//! conv input's channels, the SwiGLU halves, the residual adds and the ldy of
//! every trellis call the layer body makes are what the comparison holds.
//!
//! The 16 GB artifact is loaded the way every 27B load is -- its scope bound
//! by its own body (`ignis_artifact::bind_text_scope_27b`), its body bound by
//! its head in the leaf -- and one GDN layer (0) and two GQA layers (3, 27)
//! run for one token and for four tokens one at a time (KV, positions and the
//! GDN state carrying across them), from a residual of a hidden state's size,
//! so the residual adds carry weight.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

#[path = "support/bf16.rs"]
mod bf16;

use std::os::raw::c_void;
use std::path::{Path, PathBuf};

use ignis_artifact::fixture::{FixtureObject, build_file};
use ignis_artifact::{
    bind_text_scope_27b, f64_reference, materialize, trellis_geometry, CudaDevice, Device, Object, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gdn_layer::run_gdn_layer;
use ignis_core::gpu_profile;
use ignis_core::gqa_layer::run_gqa_layer;
use ignis_core::model_load::{load_qwen38_27b, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::trellis::{self, TrellisWeight};

use bf16::{bf16_to_f64, f32_to_bf16};


const HIDDEN: usize = 5120;
const MAX_CONTEXT_TOKENS: u32 = 128;
/// The bound on a token's layer output, relative L2 against the f64 oracle
/// over the same decoded weights (held in BF16 for it). Measured 2026-10-23
/// on the 5090, printed per token below: 0.0018 to 0.0027 on the GDN layer,
/// 0.0039 to 0.0047 on the GQA ones. Two wiring errors, built and measured the
/// same day: the q/gate split's halves swapped moved the GQA layers to 0.78 to
/// 1.0; the trellis op's workspace left dirty by a narrower call (fixed in
/// kernel/src/trellis_linear_internal.cuh) moved every layer to 0.17 to 0.86.
const LAYER_TOLERANCE: f64 = 0.01;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f32::from(bits & 0x3FF);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 => if mantissa == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A residual of a hidden state's size: varied, a few tenths per feature.
fn residual_bf16(tokens: usize) -> Vec<u16> {
    (0..tokens * HIDDEN)
        .map(|i| {
            let token = i / HIDDEN;
            let feature = i % HIDDEN;
            let seed = (token * 13 + feature) as f32;
            f32_to_bf16(seed.sin() * 0.25 * (1.0 + (feature % 997) as f32 * 1e-3))
        })
        .collect()
}

/// The 128-wide Sylvester Hadamard (unnormalized) over a row's block
/// `values[start..start + 128]`.
fn hadamard_in_row(values: &mut [f32], start: usize) {
    let mut h = 1;
    while h < 128 {
        for block in (start..start + 128).step_by(2 * h) {
            for i in block..block + h {
                let (x, y) = (values[i], values[i + h]);
                values[i] = x + y;
                values[i + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// The same over 128 whole rows of `width` from `row0`, each column on its
/// own: butterflies of rows.
fn hadamard_across_rows(values: &mut [f32], row0: usize, width: usize) {
    let mut h = 1;
    while h < 128 {
        for block in (row0..row0 + 128).step_by(2 * h) {
            for r in block..block + h {
                let (top, bottom) = values.split_at_mut((r + h) * width);
                let (a, b) = (&mut top[r * width..(r + 1) * width], &mut bottom[..width]);
                for (x, y) in a.iter_mut().zip(b.iter_mut()) {
                    let (p, q) = (*x, *y);
                    *x = p + q;
                    *y = p - q;
                }
            }
        }
        h *= 2;
    }
}

/// `name`'s trellis matrix decoded and rotated back: its served plane,
/// `[out][in]` row-major in f32 (the plane's element (r, c) is W[c][r]).
fn served_plane(device: &mut CudaDevice, reader: &Reader, name: &str) -> (Vec<f32>, usize, usize) {
    let Some(Object::Tensor(t)) = reader.find(name) else { panic!("{name}: not a tensor") };
    let g = trellis_geometry(t.format, &t.shape).unwrap_or_else(|e| panic!("{name}: {e}"));
    let payload = reader.payload(name).unwrap_or_else(|e| panic!("{name}: {e}")).data;
    let (out, inp) = (g.rows as usize, g.columns as usize);
    let record = device.allocate(payload.len() as u64).unwrap();
    device.copy_h2d(&record, 0, payload).unwrap();
    let inner = device.allocate((inp * out * 2) as u64).unwrap();
    device.synchronize().unwrap();
    let base = record.base_ptr();
    let w = TrellisWeight {
        trellis: base as *const c_void,
        suh: base.wrapping_add(g.suh_offset as usize) as *const c_void,
        svh: base.wrapping_add(g.svh_offset as usize) as *const c_void,
        k2: g.k2 as u32,
        in_features: inp as u32,
        out_features: out as u32,
        tile_stride: 0,
    };
    // The op itself on one activation, to hold the plane built below to: a
    // plane rotated wrong would otherwise read as a layer wired wrong.
    let probe: Vec<u16> = residual_bf16(1).iter().cycle().take(inp).copied().collect();
    let probe_bytes: Vec<u8> = probe.iter().flat_map(|b| b.to_le_bytes()).collect();
    let d_probe = device.allocate(probe_bytes.len() as u64).unwrap();
    device.copy_h2d(&d_probe, 0, &probe_bytes).unwrap();
    let d_y = device.allocate((out * 2) as u64).unwrap();
    let need = trellis::workspace_bytes(trellis::Call { in_features: inp as u32, out_features: out as u32, weights: 1 }, 1)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let d_ws = device.allocate(need).unwrap();
    device.synchronize().unwrap();
    let ws = trellis::TrellisWorkspace { base: d_ws.base_ptr() as *mut c_void, bytes: need };
    unsafe {
        trellis::reconstruct(&w, inner.base_ptr() as *mut c_void, std::ptr::null_mut())
            .and_then(|_| trellis::workspace_init(&ws, std::ptr::null_mut()))
            .and_then(|_| {
                trellis::linear(
                    &w,
                    d_probe.base_ptr() as *const c_void,
                    1,
                    d_y.base_ptr() as *mut c_void,
                    out as u32,
                    trellis::STORE,
                    &ws,
                    std::ptr::null_mut(),
                )
            })
    }
    .and_then(|_| ignis_core::moe::stream_sync(std::ptr::null_mut()))
    .unwrap_or_else(|e| panic!("{name}: reconstruct / probe: {e}"));
    let mut bytes = vec![0u8; inp * out * 2];
    device.copy_d2h(&inner, 0, &mut bytes).unwrap();
    let mut y_bytes = vec![0u8; out * 2];
    device.copy_d2h(&d_y, 0, &mut y_bytes).unwrap();
    device.synchronize().unwrap();
    for buffer in [record, inner, d_probe, d_y, d_ws] {
        device.deallocate(buffer).unwrap();
    }
    let half = |at: usize| u16::from_le_bytes([payload[at], payload[at + 1]]);
    let suh: Vec<f32> = (0..inp).map(|i| f16_to_f32(half(g.suh_offset as usize + 2 * i))).collect();
    let svh: Vec<f32> = (0..out).map(|j| f16_to_f32(half(g.svh_offset as usize + 2 * j))).collect();
    // P = W^T = diag(svh) H inner^T H diag(suh), H symmetric: transpose the
    // inner [in][out] matrix, rotate every row's 128-blocks (the input side)
    // and every 128-row block's columns (the output side).
    let mut plane = vec![0f32; inp * out];
    for i in 0..inp {
        for j in 0..out {
            let at = 2 * (i * out + j);
            plane[j * inp + i] = f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
        }
    }
    for row in 0..out {
        for block in 0..inp / 128 {
            hadamard_in_row(&mut plane, row * inp + block * 128);
        }
    }
    for block in 0..out / 128 {
        hadamard_across_rows(&mut plane, block * 128, inp);
    }
    for j in 0..out {
        for i in 0..inp {
            plane[j * inp + i] *= svh[j] * suh[i] / 128.0;
        }
    }
    let (mut error, mut norm) = (0f64, 0f64);
    for j in 0..out {
        let host: f64 =
            (0..inp).map(|i| f64::from(plane[j * inp + i]) * bf16_to_f64(probe[i])).sum();
        let op = bf16_to_f64(u16::from_le_bytes([y_bytes[2 * j], y_bytes[2 * j + 1]]));
        error += (op - host).powi(2);
        norm += host.powi(2);
    }
    let relative = (error / norm.max(1e-300)).sqrt();
    assert!(relative < 0.01, "{name}: the rotated plane is {relative} from the op's own product");
    (plane, out, inp)
}

/// The layer's tensors under the served artifact's names, for the oracle.
struct OracleArtifact {
    objects: Vec<FixtureObject>,
    payload: Vec<u8>,
}

impl OracleArtifact {
    fn push(&mut self, name: &str, shape: Vec<u64>, format: &'static str, bytes: &[u8]) {
        let offset = (self.payload.len() as u64).div_ceil(256) * 256;
        self.payload.resize(offset as usize, 0);
        self.payload.extend_from_slice(bytes);
        let name: &'static str = Box::leak(name.to_owned().into_boxed_str());
        self.objects.push(FixtureObject::Tensor {
            name,
            shape,
            format,
            layout: "contiguous-le-v1",
            offset,
            bytes: bytes.len() as u64,
        });
    }

    /// `rows` of served-plane f32 values, as one BF16 plane.
    fn push_plane(&mut self, name: &str, rows: &[&[f32]], cols: usize) {
        let count: usize = rows.iter().map(|r| r.len()).sum();
        let bytes: Vec<u8> = rows.iter().flat_map(|r| r.iter()).flat_map(|&v| f32_to_bf16(v).to_le_bytes()).collect();
        self.push(name, vec![(count / cols) as u64, cols as u64], "BF16", &bytes);
    }

    /// `name` copied from the trellis artifact as stored.
    fn copy(&mut self, reader: &Reader, name: &str) {
        let Some(Object::Tensor(t)) = reader.find(name) else { panic!("{name}: not a tensor") };
        let format = match t.format.name() {
            "BF16" => "BF16",
            "FP32" => "FP32",
            other => panic!("{name}: {other} is not a copied tensor's format"),
        };
        self.push(name, t.shape.clone(), format, reader.payload(name).unwrap().data);
    }
}

/// The oracle's artifact for layer `layer` of the trellis artifact, written
/// beside the test binary's other temporaries; removed on drop.
struct Written(PathBuf);

impl Drop for Written {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn oracle_artifact(device: &mut CudaDevice, reader: &Reader, layer: u32, gqa: bool) -> Written {
    let p = format!("text/layers/{layer}/");
    let mut a = OracleArtifact { objects: Vec::new(), payload: Vec::new() };
    for name in ["input_norm", "post_attention_norm"] {
        a.copy(reader, &format!("{p}{name}"));
    }
    let (gate, _, _) = served_plane(device, reader, &format!("{p}mlp/gate_proj"));
    let (up, _, _) = served_plane(device, reader, &format!("{p}mlp/up_proj"));
    a.push_plane(&format!("{p}mlp/gate_up"), &[&gate, &up], HIDDEN);
    drop((gate, up));
    let (down, _, inner) = served_plane(device, reader, &format!("{p}mlp/down"));
    a.push_plane(&format!("{p}mlp/down"), &[&down], inner);
    if gqa {
        let (q_gate, _, _) = served_plane(device, reader, &format!("{p}attention/q_proj"));
        let (k, _, _) = served_plane(device, reader, &format!("{p}attention/k_proj"));
        let (v, _, _) = served_plane(device, reader, &format!("{p}attention/v_proj"));
        // q_proj's rows: head h's q is rows [512 h, 512 h + 256), its gate
        // the next 256; the served plane is q | k | gate | v, head-major.
        let half = |second: usize| -> Vec<f32> {
            (0..24).flat_map(|h| q_gate[(h * 512 + second) * HIDDEN..(h * 512 + second + 256) * HIDDEN].to_vec()).collect()
        };
        let (q, gate) = (half(0), half(256));
        a.push_plane(&format!("{p}attention/query_key_gate_value"), &[&q, &k, &gate, &v], HIDDEN);
        for name in ["attention/query_norm", "attention/key_norm"] {
            a.copy(reader, &format!("{p}{name}"));
        }
        let (out, _, inner) = served_plane(device, reader, &format!("{p}attention/output"));
        a.push_plane(&format!("{p}attention/output"), &[&out], inner);
    } else {
        for name in ["gdn/a_log", "gdn/dt_bias", "gdn/convolution", "gdn/a_b_projection", "gdn/norm"] {
            a.copy(reader, &format!("{p}{name}"));
        }
        let (qkv, _, _) = served_plane(device, reader, &format!("{p}gdn/in_proj_qkv"));
        let (z, _, _) = served_plane(device, reader, &format!("{p}gdn/in_proj_z"));
        a.push_plane(&format!("{p}gdn/query_key_value_z"), &[&qkv, &z], HIDDEN);
        let (out, _, inner) = served_plane(device, reader, &format!("{p}gdn/output"));
        a.push_plane(&format!("{p}gdn/output"), &[&out], inner);
    }
    let path = std::env::temp_dir().join(format!("ignis-trellis-oracle-{layer}-{}.ninfer", std::process::id()));
    std::fs::write(&path, build_file(&a.objects, &a.payload)).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    Written(path)
}

#[allow(clippy::too_many_arguments)]
fn check_layer(
    oracle: &Reader,
    model: &Model,
    pool: &SeqPool,
    device: &mut CudaDevice,
    layer: u32,
    gqa: bool,
    tokens: usize,
    failures: &mut Vec<String>,
) -> bool {
    let label = format!("layer {layer}, {tokens} x T=1");
    let sequence = pool.alloc(MAX_CONTEXT_TOKENS).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
    let input = residual_bf16(tokens);
    let bytes: Vec<u8> = input.iter().flat_map(|b| b.to_le_bytes()).collect();
    let token_bytes = HIDDEN * 2;
    let in_buf = device.allocate(token_bytes as u64).unwrap_or_else(|e| panic!("{label}: {e}"));
    let out_buf = device.allocate(token_bytes as u64).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut out = Vec::with_capacity(tokens * HIDDEN);
    for token in 0..tokens {
        device.copy_h2d(&in_buf, 0, &bytes[token * token_bytes..(token + 1) * token_bytes]).unwrap();
        // The copy runs on the device's load stream, the layer on the model's
        // (GitHub #96): land it first.
        device.synchronize().unwrap();
        let ran = if gqa {
            run_gqa_layer(model, pool, &sequence, layer, &in_buf, &out_buf, 1)
        } else {
            run_gdn_layer(model, pool, &sequence, layer, &in_buf, &out_buf, 1)
        };
        if let Err(e) = ran {
            if gpu_profile::skip_or_fail(&format!("{label}: token {token}: {e}")) {
                return true;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
        let mut back = vec![0u8; token_bytes];
        device.copy_d2h(&out_buf, 0, &mut back).unwrap();
        device.synchronize().unwrap();
        out.extend(back.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
    }
    device.deallocate(in_buf).unwrap();
    device.deallocate(out_buf).unwrap();
    let reference = f64_reference::evaluate_layer(
        oracle,
        layer as usize,
        &f64_reference::LayerInput::new(input.iter().map(|&b| bf16_to_f64(b)).collect(), (0..tokens as i32).collect())
            .unwrap(),
    )
    .unwrap_or_else(|e| panic!("{label}: the oracle: {e}"));
    for token in 0..tokens {
        let range = token * HIDDEN..(token + 1) * HIDDEN;
        let (mut error, mut norm, mut delta) = (0f64, 0f64, 0f64);
        for i in range {
            let theirs = reference.residual[i];
            error += (bf16_to_f64(out[i]) - theirs).powi(2);
            norm += theirs.powi(2);
            delta += (theirs - bf16_to_f64(input[i])).powi(2);
        }
        let relative = (error / norm.max(1e-300)).sqrt();
        println!(
            "{label}: token {token}: relative L2 {relative:.5} (of {LAYER_TOLERANCE}); |delta| / |out| {:.3}",
            (delta / norm.max(1e-300)).sqrt()
        );
        if relative > LAYER_TOLERANCE {
            failures.push(format!("{label}: token {token}: relative L2 {relative} against the oracle"));
        }
    }
    false
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_trellis_bodys_layers_match_the_f64_oracle_over_their_own_weights() {
    if !Path::new(trellis_artifact()).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {}", trellis_artifact())) {
        return;
    }
    let reader = Reader::open(Path::new(trellis_artifact())).unwrap_or_else(|e| panic!("open {}: {e}", trellis_artifact()));
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind the trellis text scope: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}"));
            return;
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("materialize the trellis text scope: {e}"));
            return;
        }
    };
    let model = load_qwen38_27b(&reader, &artifact, &handles, 128, MAX_CONTEXT_TOKENS, ignis_core::KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
    let stats = model.stats();
    println!(
        "trellis load: {} tensors, {} device bytes, trellis workspace {} bytes",
        stats.bound_tensor_count, stats.vram_bytes, stats.reserved.trellis_workspace_bytes
    );
    assert!(stats.reserved.trellis_workspace_bytes > 0);
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: 4,
            max_context_tokens: MAX_CONTEXT_TOKENS,
            slot_count: 2,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));
    // Every layer and width is measured and printed before any is judged.
    let mut failures = Vec::new();
    for (layer, gqa) in [(0u32, false), (3, true), (27, true)] {
        let written = oracle_artifact(&mut device, &reader, layer, gqa);
        let oracle = Reader::open(&written.0).unwrap_or_else(|e| panic!("open the oracle's layer {layer}: {e}"));
        for tokens in [1, 4] {
            if check_layer(&oracle, &model, &pool, &mut device, layer, gqa, tokens, &mut failures) {
                return;
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}


fn trellis_artifact() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var("IGNIS_MODEL_ARTIFACT").unwrap_or_else(|_| "models/qwen3_8_27b_trellis_sc30-v2.ninfer".into())).as_str()
}
