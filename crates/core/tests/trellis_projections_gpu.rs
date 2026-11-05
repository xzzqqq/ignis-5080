//! GitHub #339 (spec 27b-16gb/05a): the 16 GB artifact's trellis matrices are
//! the served artifact's linears, row for row, in the orders spec 01's As
//! built states -- what the 27B program's trellis body assumes when it binds
//! them (`kernel/src/model.cu`, `kernel/src/trellis_body.h`).
//!
//! Each trellis matrix runs through the op (`ignis_core::trellis::linear`, its
//! `suh` and `svh` where `trellis_geometry` puts them, as the materializer's
//! view names them) on a few deterministic activations, and each block of its
//! output is held, by cosine, to the block of the served artifact's plane it
//! stands for, evaluated in f64 from the served bytes
//! (`f64_reference::matrix_product`):
//!
//! - `gdn/in_proj_qkv` is the served `gdn/query_key_value_z`'s q | k | v rows
//!   (the conv input's channels), `gdn/in_proj_z` its z rows;
//! - `attention/q_proj` is 24 heads of [q 256 | gate 256]: each head's first
//!   half is the served q block's head, its second half the served gate
//!   block's -- and, as the control that makes the check mean something, the
//!   halves taken the other way round are not;
//! - `attention/k_proj` / `v_proj` the served k / v blocks, `mlp/gate_proj` /
//!   `up_proj` the served `mlp/gate_up`'s halves, the output and down
//!   projections and the head the served ones whole.
//!
//! Two quantizations of one BF16 matrix agree to a cosine near 1 (the trellis
//! one's reconstruction error is spec 01's 0.15 mean); a block in the wrong
//! place agrees to one near 0. The floor below is an error detector between
//! the two, not a quality number (the quality gates are spec 05b's).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

#[path = "support/bf16.rs"]
mod bf16;

use std::os::raw::c_void;
use std::path::Path;

use ignis_artifact::{f64_reference, trellis_geometry, CudaDevice, Device, DeviceBuffer, Object, Reader};
use ignis_core::gpu_profile;
use ignis_core::trellis::{self, Call, TrellisWeight, TrellisWorkspace};

use bf16::{bf16_to_f64, f32_to_bf16};


fn reference_artifact() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var("IGNIS_REFERENCE_ARTIFACT")
        .unwrap_or_else(|_| "models/qwen3_8_27b_nvfp4full-v2.ninfer".into()))
}
const HIDDEN: usize = 5120;
/// Tokens per projection: two activations, so a token-major mix-up shows too.
const TOKENS: usize = 2;
/// The cosine a block must reach against the served block it stands for. Two
/// quantizations of the same rows measured 0.928 to 0.997 on 2026-10-23 (the
/// test prints each; layer 0's MLP, where spec 01 found the quant's largest
/// reconstruction error, is the lowest); a block of other rows is near 0.
const COSINE_FLOOR: f64 = 0.9;
/// The control: a block against rows it does not stand for stays below this.
const UNRELATED_CEILING: f64 = 0.3;

/// `tokens` deterministic activations of `width`, as BF16 bits, token-major,
/// of the size a normalized residual has.
fn activations(seed: usize, tokens: usize, width: usize) -> Vec<u16> {
    (0..tokens * width)
        .map(|i| {
            let x = ((i * 2654435761 + seed * 40503) % 1_000_003) as f32 / 1_000_003.0;
            f32_to_bf16((x * 2.0 - 1.0) * (1.0 + (i % 13) as f32 * 0.05))
        })
        .collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb).max(1e-300)
}

struct Gpu {
    device: CudaDevice,
}

impl Gpu {
    fn upload(&mut self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = self.device.allocate(bytes.len() as u64).unwrap_or_else(|e| panic!("allocate: {e}"));
        self.device.copy_h2d(&buffer, 0, bytes).unwrap_or_else(|e| panic!("upload: {e}"));
        self.device.synchronize().unwrap_or_else(|e| panic!("sync: {e}"));
        buffer
    }

    fn download(&mut self, buffer: &DeviceBuffer) -> Vec<u8> {
        let mut out = vec![0u8; buffer.len() as usize];
        self.device.copy_d2h(buffer, 0, &mut out).unwrap_or_else(|e| panic!("download: {e}"));
        self.device.synchronize().unwrap_or_else(|e| panic!("sync: {e}"));
        out
    }

    /// `name`'s trellis matrix applied to `x` (`[tokens][cols]` BF16): its
    /// `[tokens][rows]` output in f64, and its rows and columns.
    fn trellis_product(&mut self, reader: &Reader, name: &str, x: &[u16], tokens: usize) -> Option<(Vec<f64>, usize)> {
        let Some(Object::Tensor(t)) = reader.find(name) else { panic!("{name}: not a tensor of the trellis artifact") };
        let g = trellis_geometry(t.format, &t.shape).unwrap_or_else(|e| panic!("{name}: {e}"));
        let record = self.upload(reader.payload(name).unwrap_or_else(|e| panic!("{name}: {e}")).data);
        let base = record.base_ptr();
        let (rows, cols) = (g.rows as u32, g.columns as u32);
        let w = TrellisWeight {
            trellis: base as *const c_void,
            suh: base.wrapping_add(g.suh_offset as usize) as *const c_void,
            svh: base.wrapping_add(g.svh_offset as usize) as *const c_void,
            k2: g.k2 as u32,
            in_features: cols,
            out_features: rows,
            tile_stride: 0,
        };
        let bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
        let d_x = self.upload(&bytes);
        let d_y = self.upload(&vec![0u8; tokens * rows as usize * 2]);
        let need = trellis::workspace_bytes(Call { in_features: cols, out_features: rows, weights: 1 }, tokens as u32)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let d_ws = self.upload(&vec![0u8; need as usize]);
        let ws = TrellisWorkspace { base: d_ws.base_ptr() as *mut c_void, bytes: need };
        let ran = unsafe {
            trellis::workspace_init(&ws, std::ptr::null_mut()).and_then(|_| {
                trellis::linear(
                    &w,
                    d_x.base_ptr() as *const c_void,
                    tokens as u32,
                    d_y.base_ptr() as *mut c_void,
                    rows,
                    trellis::STORE,
                    &ws,
                    std::ptr::null_mut(),
                )
            })
        };
        // The call ran on the legacy stream; the device copies on its own.
        if let Err(e) = ran.and_then(|_| ignis_core::moe::stream_sync(std::ptr::null_mut())) {
            gpu_profile::skip_or_fail(&format!("{name}: ignis_trellis_linear: {e}"));
            return None;
        }
        let y = self.download(&d_y);
        for buffer in [record, d_x, d_y, d_ws] {
            self.device.deallocate(buffer).unwrap_or_else(|e| panic!("free: {e}"));
        }
        Some((y.chunks_exact(2).map(|b| bf16_to_f64(u16::from_le_bytes([b[0], b[1]]))).collect(), rows as usize))
    }
}

/// The served plane `name` (`[rows, cols]`) applied to each token of `x`, in
/// f64: `[tokens][rows]`.
fn served_product(served: &Reader, name: &str, rows: usize, cols: usize, x: &[u16], tokens: usize) -> Vec<f64> {
    (0..tokens)
        .flat_map(|t| {
            let input: Vec<f64> = x[t * cols..(t + 1) * cols].iter().map(|&b| bf16_to_f64(b)).collect();
            f64_reference::matrix_product(served, name, rows, cols, &input).unwrap_or_else(|e| panic!("{name}: {e}"))
        })
        .collect()
}

/// Row `r` of every token of a `[tokens][width]` output, gathered by `rows`.
fn gather(out: &[f64], width: usize, tokens: usize, rows: impl Fn(usize) -> usize + Copy, count: usize) -> Vec<f64> {
    (0..tokens).flat_map(|t| (0..count).map(move |i| out[t * width + rows(i)])).collect()
}

/// One block's cosine, printed and held to the floor.
fn holds(label: &str, ours: &[f64], theirs: &[f64]) -> f64 {
    let c = cosine(ours, theirs);
    println!("{label}: cosine {c:.5}");
    assert!(c >= COSINE_FLOOR, "{label}: cosine {c} below {COSINE_FLOOR}: the block is not the served one's rows");
    c
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_trellis_matrices_are_the_served_linears_in_spec_01s_row_orders() {
    for path in [trellis_artifact(), reference_artifact()] {
        if !Path::new(path).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {path}")) {
            return;
        }
    }
    let ours = Reader::open(Path::new(trellis_artifact())).unwrap_or_else(|e| panic!("open {}: {e}", trellis_artifact()));
    let served = Reader::open(Path::new(reference_artifact())).unwrap_or_else(|e| panic!("open {}: {e}", reference_artifact()));
    let device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return;
        }
    };
    if let Err(e) = trellis::prepare() {
        gpu_profile::skip_or_fail(&format!("ignis_trellis_prepare: {e}"));
        return;
    }
    let mut gpu = Gpu { device };
    let (q_width, kv_width, ffn, state_rows) = (24 * 256, 4 * 256, 17_408, 6144);

    // A GDN layer (0, whose served output projection is NVFP4) and the GDN
    // layer whose served output projection is the BF16 exception (4).
    for layer in [0usize, 4] {
        let p = format!("text/layers/{layer}/");
        let x = activations(layer, TOKENS, HIDDEN);
        let Some((qkv, _)) = gpu.trellis_product(&ours, &format!("{p}gdn/in_proj_qkv"), &x, TOKENS) else { return };
        let Some((z, _)) = gpu.trellis_product(&ours, &format!("{p}gdn/in_proj_z"), &x, TOKENS) else { return };
        let qkvz = served_product(&served, &format!("{p}gdn/query_key_value_z"), 16_384, HIDDEN, &x, TOKENS);
        for (block, at, width) in [("q", 0, 2048), ("k", 2048, 2048), ("v", 4096, 6144)] {
            holds(
                &format!("layer {layer} gdn/in_proj_qkv {block}"),
                &gather(&qkv, 10_240, TOKENS, |i| at + i, width),
                &gather(&qkvz, 16_384, TOKENS, |i| at + i, width),
            );
        }
        holds(
            &format!("layer {layer} gdn/in_proj_z"),
            &z,
            &gather(&qkvz, 16_384, TOKENS, |i| 10_240 + i, state_rows),
        );
        let xo = activations(layer + 100, TOKENS, state_rows);
        let Some((out, _)) = gpu.trellis_product(&ours, &format!("{p}gdn/output"), &xo, TOKENS) else { return };
        holds(
            &format!("layer {layer} gdn/output"),
            &out,
            &served_product(&served, &format!("{p}gdn/output"), HIDDEN, state_rows, &xo, TOKENS),
        );
    }

    // The GQA layers: 3 (the served fused projection is BF16), 27 (NVFP4).
    for layer in [3usize, 27] {
        let p = format!("text/layers/{layer}/");
        let x = activations(layer, TOKENS, HIDDEN);
        let Some((q_gate, _)) = gpu.trellis_product(&ours, &format!("{p}attention/q_proj"), &x, TOKENS) else { return };
        let Some((k, _)) = gpu.trellis_product(&ours, &format!("{p}attention/k_proj"), &x, TOKENS) else { return };
        let Some((v, _)) = gpu.trellis_product(&ours, &format!("{p}attention/v_proj"), &x, TOKENS) else { return };
        let qkgv =
            served_product(&served, &format!("{p}attention/query_key_gate_value"), 14_336, HIDDEN, &x, TOKENS);
        // The served blocks: q | k | gate | v, each head-major.
        let (served_q, served_k, served_gate, served_v) =
            (0, q_width, q_width + kv_width, 2 * q_width + kv_width);
        // q_proj's head h is rows [512 h, 512 h + 256) of q and then of gate.
        let ours_q = gather(&q_gate, 2 * q_width, TOKENS, |i| (i / 256) * 512 + i % 256, q_width);
        let ours_gate = gather(&q_gate, 2 * q_width, TOKENS, |i| (i / 256) * 512 + 256 + i % 256, q_width);
        let theirs_q = gather(&qkgv, 14_336, TOKENS, |i| served_q + i, q_width);
        let theirs_gate = gather(&qkgv, 14_336, TOKENS, |i| served_gate + i, q_width);
        holds(&format!("layer {layer} attention/q_proj q"), &ours_q, &theirs_q);
        holds(&format!("layer {layer} attention/q_proj gate"), &ours_gate, &theirs_gate);
        // The control: the halves the other way round.
        for (label, a, b) in [("q as gate", &ours_q, &theirs_gate), ("gate as q", &ours_gate, &theirs_q)] {
            let c = cosine(a, b);
            println!("layer {layer} attention/q_proj {label} (control): cosine {c:.5}");
            assert!(c < UNRELATED_CEILING, "layer {layer} {label}: {c}: the control cannot tell the halves apart");
        }
        holds(&format!("layer {layer} attention/k_proj"), &k, &gather(&qkgv, 14_336, TOKENS, |i| served_k + i, kv_width));
        holds(&format!("layer {layer} attention/v_proj"), &v, &gather(&qkgv, 14_336, TOKENS, |i| served_v + i, kv_width));
        let xo = activations(layer + 100, TOKENS, q_width);
        let Some((out, _)) = gpu.trellis_product(&ours, &format!("{p}attention/output"), &xo, TOKENS) else { return };
        holds(
            &format!("layer {layer} attention/output"),
            &out,
            &served_product(&served, &format!("{p}attention/output"), HIDDEN, q_width, &xo, TOKENS),
        );
    }

    // The MLP of a GDN and of a GQA layer.
    for layer in [0usize, 27] {
        let p = format!("text/layers/{layer}/");
        let x = activations(layer + 200, TOKENS, HIDDEN);
        let Some((gate, _)) = gpu.trellis_product(&ours, &format!("{p}mlp/gate_proj"), &x, TOKENS) else { return };
        let Some((up, _)) = gpu.trellis_product(&ours, &format!("{p}mlp/up_proj"), &x, TOKENS) else { return };
        let gate_up = served_product(&served, &format!("{p}mlp/gate_up"), 2 * ffn, HIDDEN, &x, TOKENS);
        holds(&format!("layer {layer} mlp/gate_proj"), &gate, &gather(&gate_up, 2 * ffn, TOKENS, |i| i, ffn));
        holds(&format!("layer {layer} mlp/up_proj"), &up, &gather(&gate_up, 2 * ffn, TOKENS, |i| ffn + i, ffn));
        let xd = activations(layer + 300, TOKENS, ffn);
        let Some((down, _)) = gpu.trellis_product(&ours, &format!("{p}mlp/down"), &xd, TOKENS) else { return };
        holds(
            &format!("layer {layer} mlp/down"),
            &down,
            &served_product(&served, &format!("{p}mlp/down"), HIDDEN, ffn, &xd, TOKENS),
        );
    }

    // The head, at one token: the served one is W8.
    let x = activations(400, 1, HIDDEN);
    let Some((logits, vocab)) = gpu.trellis_product(&ours, "text/output_head", &x, 1) else { return };
    assert_eq!(vocab, 248_320);
    holds("text/output_head", &logits, &served_product(&served, "text/output_head", vocab, HIDDEN, &x, 1));
}

fn trellis_artifact() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var("IGNIS_MODEL_ARTIFACT").unwrap_or_else(|_| "models/qwen3_8_27b_trellis_sc30-v2.ninfer".into())).as_str()
}
