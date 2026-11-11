// The dense trellis linear against the fp64 product of its decoded weights -- OURS (spec
// 27b-16gb/02 Acceptance 2 and 3, GitHub #337).
//
// Every comparison uses the header's tolerance, |y - Y| <= 2^-8 |Y| + 2^-9 s (trellis_test_common.h
// computes Y and s), and prints the share of it the worst output used.
//
//   fixtures     the six exllamav3-encoded projections (K = 2 .. 6) with the activations and fp64
//                outputs the recorder wrote: our fp64 reference reproduces them, and both routes
//                (3 tokens on decode; the same 3 tokens tiled to 130 on prefill) match them.
//   shapes       every shape the 27B serves, fused and split, each at a K of the quant (every K
//                covered), over counter-hash words, at 1, 2, 8, 16, 64 and 1,024 tokens. Large
//                token counts are checked on 128 rows spread over every 128-token tile.
//   epilogues    the residual add and SwiGLU (gate and up of different K) on both routes.
//   fp16 cap     both routes' fp16 accumulation at in = 17,408 with every product of every fold
//                at the activation scale's bound (a constant weight at the codebook's extreme,
//                activations that rotate to a constant): no overflow, still inside tolerance.
//   determinism  the same call twice, twice under the SM cap at 36 (the prefill also against
//                uncapped: it does not split), and replayed three times from a CUDA graph captured
//                in global mode (which refuses an allocation): the same bits every time.
//   shared workspace  one workspace through a load's calls of five shapes in turn, both routes:
//                each call's bits are its own fresh workspace's (GitHub #339).
//   T=8 MLP     same-K gate/up at the real shape, plus token/bit-width/SM-cap fallbacks, against
//                fp64; ten different calls share one workspace through five graph replays.
//   rejects      the arguments the ops refuse.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU or fixture fails.

#include "trellis_test_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace trellis_test;

namespace {

std::string fixture(const std::string &name) { return std::string(IGNIS_TRELLIS_DENSE_FIXTURE_DIR) + "/" + name; }

// A trellis weight on the device: counter-hash words, hash scales, and its decoded inner matrix.
struct DeviceWeight {
  uint32_t k2 = 0, in = 0, out = 0;
  DeviceBytes words, suh, svh, inner;
  std::vector<uint16_t> suh_h, svh_h;
  ignis_trellis_weight w{};
};

// Scale vectors like a quant's: random sign, magnitude in [0.5, 2) times `amp`.
std::vector<uint16_t> hash_scales(uint32_t stream, int n, float amp) {
  std::vector<uint16_t> v(n);
  for (int i = 0; i < n; ++i) {
    const uint32_t h = moe_test::hash_u32(stream, static_cast<uint64_t>(i));
    const float mag = amp * (0.5f + 1.5f * static_cast<float>(h >> 9) * 0x1p-23f);
    v[i] = moe_test::f32_to_f16((h & 1u) ? -mag : mag);
  }
  return v;
}

void make_weight(DeviceWeight &d, uint32_t k2, uint32_t in, uint32_t out, uint32_t stream) {
  d.k2 = k2;
  d.in = in;
  d.out = out;
  const uint64_t n_words = static_cast<uint64_t>(in / 16) * (out / 16) * tile_words16(k2);
  d.words = DeviceBytes(n_words * 2);
  device_hash_words(d.words.p, n_words, stream);
  d.suh_h = hash_scales(stream + 1, static_cast<int>(in), 0.05f);
  d.svh_h = hash_scales(stream + 2, static_cast<int>(out), 0.03f);
  d.suh = DeviceBytes(in * 2);
  d.svh = DeviceBytes(out * 2);
  upload(d.suh, d.suh_h);
  upload(d.svh, d.svh_h);
  d.w = ignis_trellis_weight{d.words.p, d.suh.p, d.svh.p, k2, in, out, 0};
  d.inner = DeviceBytes(static_cast<size_t>(in) * out * 2);
  TL_RC(ignis_trellis_reconstruct(&d.w, d.inner.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
}

// BF16 activations with a few loud channels, as a residual stream has.
std::vector<uint16_t> make_x(uint32_t stream, int tokens, int in) {
  std::vector<uint16_t> x(static_cast<size_t>(tokens) * in);
  for (size_t i = 0; i < x.size(); ++i) {
    float v = moe_test::hash_uniform(stream, i, 1.5f);
    if ((i % static_cast<size_t>(in)) % 97 == 13) v *= 24.0f;
    x[i] = moe_test::f32_to_bf16(v);
  }
  return x;
}

// The rows a large call is checked on: all of them up to 64, else 16 rows of every 128-row tile
// at a tile-dependent offset (the last tile's ragged rows included).
std::vector<int> checked_rows(int tokens) {
  std::vector<int> rows;
  if (tokens <= 64) {
    for (int t = 0; t < tokens; ++t) rows.push_back(t);
    return rows;
  }
  for (int t0 = 0; t0 < tokens; t0 += 128) {
    const int span = std::min(128, tokens - t0);
    for (int i = 0; i < 16 && i < span; ++i) rows.push_back(t0 + (i * 8 + (t0 / 128) * 3) % span);
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

std::vector<uint16_t> pick_rows(const std::vector<uint16_t> &m, const std::vector<int> &rows, int width, int ld) {
  std::vector<uint16_t> o(rows.size() * static_cast<size_t>(width));
  for (size_t r = 0; r < rows.size(); ++r)
    std::copy_n(m.begin() + static_cast<size_t>(rows[r]) * ld, width, o.begin() + r * width);
  return o;
}

struct Workspace {
  DeviceBytes mem;
  ignis_trellis_workspace ws{};
  explicit Workspace(uint64_t bytes) : mem(bytes) {
    ws.base = mem.p;
    ws.bytes = bytes;
    TL_RC(ignis_trellis_workspace_init(&ws, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
};

double g_worst_decode = 0.0, g_worst_prefill = 0.0;

void note(int tokens, double worst) {
  double &w = tokens <= IGNIS_TRELLIS_DECODE_MAX_TOKENS ? g_worst_decode : g_worst_prefill;
  w = std::max(w, worst);
}

// y = x W through `epilogue` on `tokens` tokens, against the reference, with residual r when ADD.
void check_linear(const std::string &label, const DeviceWeight &d, int tokens, const Workspace &ws, uint32_t epilogue,
                  uint32_t stream) {
  const int in = static_cast<int>(d.in), out = static_cast<int>(d.out);
  const std::vector<uint16_t> x = make_x(stream, tokens, in);
  std::vector<uint16_t> r(static_cast<size_t>(tokens) * out, 0);
  if (epilogue == IGNIS_TRELLIS_ADD) {
    for (size_t i = 0; i < r.size(); ++i) r[i] = moe_test::f32_to_bf16(moe_test::hash_uniform(stream + 7, i, 0.8f));
  }
  DeviceBytes d_x(x.size() * 2), d_y(r.size() * 2);
  upload(d_x, x);
  upload(d_y, r);
  TL_RC(ignis_trellis_linear(&d.w, d_x.p, tokens, d_y.p, out, epilogue, &ws.ws, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<uint16_t> y = download<uint16_t>(d_y.p, r.size());
  const std::vector<int> rows = checked_rows(tokens);
  const Reference ref = reference(pick_rows(x, rows, in, in), static_cast<int>(rows.size()), d.inner.p, in, out, d.suh_h, d.svh_h);
  std::vector<double> want = ref.y, got = bf16_to_f64(pick_rows(y, rows, out, out)), bounds(want.size());
  const std::vector<double> res = bf16_to_f64(pick_rows(r, rows, out, out));
  for (size_t i = 0; i < want.size(); ++i) {
    want[i] += res[i];
    bounds[i] = bound(want[i], ref.s[i]);
  }
  const Agreement a = agree(got, want, bounds);
  note(tokens, a.worst);
  check(a.bad == 0, label + ": " + std::to_string(a.bad) + " outputs outside the tolerance (first at " +
                        std::to_string(a.first_bad) + ", worst " + std::to_string(a.worst) + " of the bound)");
  std::printf("  %-34s %5d tok: worst %.3f of the bound, rel rms %.2e\n", label.c_str(), tokens, a.worst, a.rel_rms);
}

// h = silu(x G) * (x U): the bound composes the two outputs' through silu, plus the prefill's BF16
// gate between its passes.
std::vector<uint16_t> check_swiglu(const std::string &label, const DeviceWeight &g, const DeviceWeight &u, int tokens,
                                 const Workspace &ws, uint32_t stream) {
  const int in = static_cast<int>(g.in), out = static_cast<int>(g.out);
  const std::vector<uint16_t> x = make_x(stream, tokens, in);
  DeviceBytes d_x(x.size() * 2), d_h(static_cast<size_t>(tokens) * out * 2);
  upload(d_x, x);
  MOE_CUDA(cudaMemset(d_h.p, 0xff, d_h.bytes));
  TL_RC(ignis_trellis_linear_swiglu(&g.w, &u.w, d_x.p, tokens, d_h.p, out, &ws.ws, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<uint16_t> h = download<uint16_t>(d_h.p, static_cast<size_t>(tokens) * out);
  const std::vector<int> rows = checked_rows(tokens);
  const std::vector<uint16_t> xs = pick_rows(x, rows, in, in);
  const Reference rg = reference(xs, static_cast<int>(rows.size()), g.inner.p, in, out, g.suh_h, g.svh_h);
  const Reference ru = reference(xs, static_cast<int>(rows.size()), u.inner.p, in, out, u.suh_h, u.svh_h);
  const bool prefill = tokens > IGNIS_TRELLIS_DECODE_MAX_TOKENS;
  std::vector<double> want(rg.y.size()), bounds(rg.y.size());
  for (size_t i = 0; i < want.size(); ++i) {
    const double G = rg.y[i], U = ru.y[i];
    const double sig = 1.0 / (1.0 + std::exp(-G));
    const double silu = G * sig;
    const double dsilu = sig * (1.0 + G * (1.0 - sig));
    want[i] = silu * U;
    const double eg = std::ldexp(rg.s[i], -9) + (prefill ? std::ldexp(std::fabs(G), -8) : 0.0);
    bounds[i] = std::ldexp(std::fabs(want[i]), -8) + std::fabs(dsilu * U) * eg + std::fabs(silu) * std::ldexp(ru.s[i], -9);
  }
  const Agreement a = agree(bf16_to_f64(pick_rows(h, rows, out, out)), want, bounds);
  note(tokens, a.worst);
  check(a.bad == 0, label + ": " + std::to_string(a.bad) + " outputs outside the tolerance (worst " +
                        std::to_string(a.worst) + " of the bound)");
  std::printf("  %-34s %5d tok: worst %.3f of the bound, rel rms %.2e\n", label.c_str(), tokens, a.worst, a.rel_rms);
  return h;
}

const char *kName[] = {"k2", "k2p5", "k3", "k4", "k5", "k6"};
const uint32_t kK2[] = {4, 5, 6, 8, 10, 12};

void fixtures_arm() {
  for (int k = 0; k < 6; ++k) {
    const auto fx = moe_test::read_fixture(fixture(std::string("trellis_") + kName[k] + ".bin"));
    const auto &rc = moe_test::need(fx, "reconstruct");
    const int in = static_cast<int>(rc.dims[0]), out = static_cast<int>(rc.dims[1]);
    const auto words = moe_test::need(fx, "trellis").as<uint16_t>();
    const auto suh = moe_test::need(fx, "suh").as<uint16_t>();
    const auto svh = moe_test::need(fx, "svh").as<uint16_t>();
    const auto xf = moe_test::need(fx, "x").as<float>();
    const auto y64 = moe_test::need(fx, "y64").as<double>();
    const int tokens = static_cast<int>(moe_test::need(fx, "x").dims[0]);
    DeviceBytes d_words(words.size() * 2), d_suh(suh.size() * 2), d_svh(svh.size() * 2), d_inner(static_cast<size_t>(in) * out * 2);
    upload(d_words, words);
    upload(d_suh, suh);
    upload(d_svh, svh);
    const ignis_trellis_weight w{d_words.p, d_suh.p, d_svh.p, kK2[k], static_cast<uint32_t>(in), static_cast<uint32_t>(out), 0};
    TL_RC(ignis_trellis_reconstruct(&w, d_inner.p, nullptr));
    std::vector<uint16_t> x(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) x[i] = moe_test::f32_to_bf16(xf[i]);  // exact: BF16 values
    const Reference ref = reference(x, tokens, d_inner.p, in, out, suh, svh);
    double ref_err = 0.0, ymax = 0.0;
    for (size_t i = 0; i < y64.size(); ++i) {
      ref_err = std::max(ref_err, std::fabs(ref.y[i] - y64[i]));
      ymax = std::max(ymax, std::fabs(y64[i]));
    }
    check(ref_err <= 1e-9 * ymax, std::string(kName[k]) + ": the fp64 reference differs from the recorded y64");
    for (const int t : {tokens, 130}) {
      std::vector<uint16_t> xt(static_cast<size_t>(t) * in);
      for (int r = 0; r < t; ++r) std::copy_n(x.begin() + static_cast<size_t>(r % tokens) * in, in, xt.begin() + static_cast<size_t>(r) * in);
      Workspace ws(ignis_trellis_workspace_bytes(in, out, 1, t));
      DeviceBytes d_x(xt.size() * 2), d_y(static_cast<size_t>(t) * out * 2);
      upload(d_x, xt);
      TL_RC(ignis_trellis_linear(&w, d_x.p, t, d_y.p, out, IGNIS_TRELLIS_STORE, &ws.ws, nullptr));
      MOE_CUDA(cudaDeviceSynchronize());
      const std::vector<double> got = bf16_to_f64(download<uint16_t>(d_y.p, static_cast<size_t>(t) * out));
      std::vector<double> want(got.size()), bounds(got.size());
      for (size_t i = 0; i < got.size(); ++i) {
        const size_t j = (i / out % tokens) * out + i % out;
        want[i] = y64[j];
        bounds[i] = bound(y64[j], ref.s[j]);
      }
      const Agreement a = agree(got, want, bounds);
      note(t, a.worst);
      check(a.bad == 0, std::string(kName[k]) + " fixture at " + std::to_string(t) + " tokens: " + std::to_string(a.bad) +
                            " outputs outside the tolerance");
      std::printf("  fixture %-5s %dx%d %3d tok: worst %.3f of the bound (reference vs y64 %.1e)\n", kName[k], in, out, t,
                  a.worst, ref_err / ymax);
    }
  }
}

struct Shape {
  const char *name;
  uint32_t in, out, k2;
};

// spec 02's table, fused and split, each at a K the quant gives it (head: K = 4), every K covered.
const Shape kShapes[] = {
    {"gate_up (fused)", 5120, 34816, 6}, {"gate", 5120, 17408, 4},   {"down", 17408, 5120, 6},
    {"qkvz (fused)", 5120, 16384, 5},    {"qkv", 5120, 10240, 6},    {"z", 5120, 6144, 8},
    {"gdn out", 6144, 5120, 8},          {"qkgv (fused)", 5120, 14336, 12}, {"q", 5120, 12288, 4},
    {"k / v", 5120, 1024, 10},           {"head", 5120, 248320, 8},
};

void shapes_arm() {
  const int tokens[] = {1, 2, 8, 16, 64, 1024};
  uint32_t stream = 50000;
  for (const Shape &s : kShapes) {
    DeviceWeight d;
    make_weight(d, s.k2, s.in, s.out, stream += 10);
    uint64_t need = 0;
    for (const int t : tokens) need = std::max(need, ignis_trellis_workspace_bytes(s.in, s.out, 1, t));
    Workspace ws(need);
    for (const int t : tokens) {
      check_linear(std::string(s.name) + " K2=" + std::to_string(s.k2), d, t, ws, IGNIS_TRELLIS_STORE, stream + t);
    }
  }
}

void epilogues_arm() {
  {
    DeviceWeight d;
    make_weight(d, 6, 17408, 5120, 61000);  // down, the residual add's widest input
    uint64_t need = 0;
    for (const int t : {1, 16, 64, 1024}) need = std::max(need, ignis_trellis_workspace_bytes(17408, 5120, 1, t));
    Workspace ws(need);
    for (const int t : {1, 16, 64, 1024}) check_linear("down + residual", d, t, ws, IGNIS_TRELLIS_ADD, 61100 + t);
  }
  {
    DeviceWeight g, u;
    make_weight(g, 6, 5120, 17408, 62000);
    make_weight(u, 4, 5120, 17408, 62100);
    uint64_t need = 0;
    for (const int t : {1, 8, 16, 64, 1024}) need = std::max(need, ignis_trellis_workspace_bytes(5120, 17408, 2, t));
    Workspace ws(need);
    for (const int t : {1, 8, 16, 64, 1024}) check_swiglu("swiglu gate K3, up K2", g, u, t, ws, 62200 + t);
  }
  {
    // A fused gate_up plane's two halves as views (tile_stride), K = 5.
    DeviceWeight plane;
    make_weight(plane, 10, 5120, 2048, 63000);
    DeviceWeight g, u;
    for (int half = 0; half < 2; ++half) {
      DeviceWeight &v = half == 0 ? g : u;
      v.k2 = 10;
      v.in = 5120;
      v.out = 1024;
      v.suh_h = plane.suh_h;
      v.svh_h.assign(plane.svh_h.begin() + half * 1024, plane.svh_h.begin() + (half + 1) * 1024);
      v.w = plane.w;
      v.w.out = 1024;
      v.w.tile_stride = 2048 / 16;
      v.w.trellis = static_cast<const char *>(plane.words.p) + static_cast<size_t>(half * 1024 / 16) * tile_words16(10) * 2;
      v.w.svh = static_cast<const char *>(plane.svh.p) + half * 1024 * 2;
      v.inner = DeviceBytes(static_cast<size_t>(5120) * 1024 * 2);
      TL_RC(ignis_trellis_reconstruct(&v.w, v.inner.p, nullptr));
    }
    MOE_CUDA(cudaDeviceSynchronize());
    uint64_t need = 0;
    for (const int t : {8, 200}) need = std::max(need, ignis_trellis_workspace_bytes(5120, 1024, 2, t));
    Workspace ws(need);
    for (const int t : {8, 200}) check_swiglu("swiglu, fused plane views K5", g, u, t, ws, 63100 + t);
  }
}

// The mul1 value of the state a tile has at every step when its stream repeats a K-bit pattern.
void fp16_cap_arm() {
  const uint32_t k2 = 8, in = 17408, out = 256;
  const int words16 = static_cast<int>(tile_words16(k2));
  // The K = 4 pattern whose constant tile decodes to the codebook's largest magnitude.
  std::vector<uint16_t> tile(words16);
  float best = 0.0f;
  std::vector<uint16_t> best_tile;
  for (uint32_t pat = 0; pat < 16; ++pat) {
    for (int i = 0; i < words16; ++i) {
      // A 32-bit word holds eight 4-bit copies; the stream is big-endian in u32 units.
      const uint32_t u32 = pat * 0x11111111u;
      tile[i] = static_cast<uint16_t>((i & 1) ? (u32 >> 16) : (u32 & 0xFFFFu));
    }
    const std::vector<uint16_t> w = moe_test::host_trellis_decode(tile.data(), k2, 16, 16);
    const float v = moe_test::f16_to_f32(w[0]);
    bool constant = true;
    for (const uint16_t e : w) constant = constant && e == w[0];
    if (constant && std::fabs(v) > std::fabs(best)) {
      best = v;
      best_tile = tile;
    }
  }
  check(std::fabs(best) > 3.0f, "a constant tile near the codebook's extreme exists");
  std::vector<uint16_t> words(static_cast<size_t>(in / 16) * (out / 16) * words16);
  for (size_t t = 0; t < words.size() / words16; ++t) std::copy(best_tile.begin(), best_tile.end(), words.begin() + t * words16);
  DeviceWeight d;
  d.k2 = k2;
  d.in = in;
  d.out = out;
  d.words = DeviceBytes(words.size() * 2);
  upload(d.words, words);
  d.suh_h.assign(in, moe_test::f32_to_f16(1.0f));
  d.svh_h = hash_scales(64000, out, 0.03f);
  d.suh = DeviceBytes(in * 2);
  d.svh = DeviceBytes(out * 2);
  upload(d.suh, d.suh_h);
  upload(d.svh, d.svh_h);
  d.w = ignis_trellis_weight{d.words.p, d.suh.p, d.svh.p, k2, in, out, 0};
  d.inner = DeviceBytes(static_cast<size_t>(in) * out * 2);
  TL_RC(ignis_trellis_reconstruct(&d.w, d.inner.p, nullptr));
  // Activations whose every 128-block rotates to a constant: one entry per block (x o suh = a
  // sqrt(128) e_0 gives H(.) = a 1). 11.25 / sqrt(128) = 0.9944 scales to 509.1, just under the
  // bound of 512, so every fold sums 32 products of 509.1 x |w| = 32 x 1,758 = 56,256 < 65,504.
  for (const int tokens : {8, 130}) {  // both routes: decode and prefill
  std::vector<uint16_t> x(static_cast<size_t>(tokens) * in, 0);
  for (int t = 0; t < tokens; ++t)
    for (uint32_t b = 0; b < in; b += 128) x[static_cast<size_t>(t) * in + b] = moe_test::f32_to_bf16(t % 2 ? -11.25f : 11.25f);
  Workspace ws(ignis_trellis_workspace_bytes(in, out, 1, tokens));
  DeviceBytes d_x(x.size() * 2), d_y(static_cast<size_t>(tokens) * out * 2);
  upload(d_x, x);
  TL_RC(ignis_trellis_linear(&d.w, d_x.p, tokens, d_y.p, out, IGNIS_TRELLIS_STORE, &ws.ws, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<uint16_t> y = download<uint16_t>(d_y.p, static_cast<size_t>(tokens) * out);
  const std::vector<int> rows = checked_rows(tokens);
  const Reference ref = reference(pick_rows(x, rows, in, in), static_cast<int>(rows.size()), d.inner.p, in, out, d.suh_h, d.svh_h);
  std::vector<double> bounds(ref.y.size());
  for (size_t i = 0; i < bounds.size(); ++i) bounds[i] = bound(ref.y[i], ref.s[i]);
  const Agreement a = agree(bf16_to_f64(pick_rows(y, rows, out, out)), ref.y, bounds);
  note(tokens, a.worst);
  check(a.bad == 0, "fp16 accumulation at its bound, in = 17,408, " + std::to_string(tokens) + " tokens: " +
                        std::to_string(a.bad) + " outputs outside the tolerance");
  std::printf("  fp16 cap: weight %.4f everywhere, in 17408, %d tok: worst %.3f of the bound, rel rms %.2e\n", best,
              tokens, a.worst, a.rel_rms);
  }
}

std::vector<uint16_t> run_bits(const DeviceWeight &g, const DeviceWeight *u, int tokens, const Workspace &ws, cudaStream_t s,
                               const DeviceBytes &d_x, DeviceBytes &d_y) {
  if (u == nullptr) {
    TL_RC(ignis_trellis_linear(&g.w, d_x.p, tokens, d_y.p, g.out, IGNIS_TRELLIS_STORE, &ws.ws, s));
  } else {
    TL_RC(ignis_trellis_linear_swiglu(&g.w, &u->w, d_x.p, tokens, d_y.p, g.out, &ws.ws, s));
  }
  MOE_CUDA(cudaStreamSynchronize(s));
  return download<uint16_t>(d_y.p, static_cast<size_t>(tokens) * g.out);
}

void set_cap(const char *value) {
#ifdef _WIN32
  _putenv_s("IGNIS_TRELLIS_SM_CAP", value);
#else
  setenv("IGNIS_TRELLIS_SM_CAP", value, 1);
#endif
  TL_RC(ignis_trellis_prepare());
}

// The mixed-K epilogue tests do not enter the 5080's T=8 MLP route. Check every selected K
// against fp64, then alternate it with fallback calls without resetting their shared workspace.
void mlp_t8_arm() {
  constexpr uint32_t in = 5120, out = 17408;
  int device = 0, sms = 0;
  MOE_CUDA(cudaGetDevice(&device));
  MOE_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  const uint32_t k2[] = {4, 6, 8, 5};
  DeviceWeight g[4], u[4];
  set_cap("0");
  for (int i = 0; i < 4; ++i) {
    make_weight(g[i], k2[i], in, out, 100000 + 100 * i);
    make_weight(u[i], k2[i], in, out, 100050 + 100 * i);
  }
  struct Call {
    const char *name;
    const DeviceWeight *gate, *up;
    int tokens;
  };
  const Call calls[] = {{"MLP K2", &g[0], &u[0], 8},        {"MLP K3", &g[1], &u[1], 8},
                        {"MLP K4", &g[2], &u[2], 8},        {"MLP K2.5 fallback", &g[3], &u[3], 8},
                        {"MLP T1 fallback", &g[1], &u[1], 1}, {"MLP K4 swapped", &u[2], &g[2], 8},
                        {"MLP K2 swapped", &u[0], &g[0], 8}, {"MLP mixed K3/K2", &g[1], &u[0], 8},
                        {"MLP T16 fallback", &g[1], &u[1], 16}, {"MLP K3 swapped", &u[1], &g[1], 8}};
  constexpr int count = sizeof(calls) / sizeof(calls[0]);
  uint64_t bytes = 0;
  for (const Call &c : calls) bytes = std::max(bytes, ignis_trellis_workspace_bytes(in, out, 2, c.tokens));
  Workspace shared(bytes);
  DeviceBytes x[count], y[count];
  std::vector<uint16_t> expected[count];
  cudaStream_t s;
  MOE_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  for (int i = 0; i < count; ++i) {
    const Call &c = calls[i];
    Workspace fresh(ignis_trellis_workspace_bytes(in, out, 2, c.tokens));
    expected[i] = check_swiglu(c.name, *c.gate, *c.up, c.tokens, fresh, 101000 + i);
    x[i] = DeviceBytes(static_cast<size_t>(c.tokens) * in * 2);
    y[i] = DeviceBytes(static_cast<size_t>(c.tokens) * out * 2);
    // upload waits for the pageable legacy-stream copy before the nonblocking stream reads it.
    upload(x[i], make_x(101000 + i, c.tokens, in));
    MOE_CUDA(cudaMemsetAsync(y[i].p, 0xff, y[i].bytes, s));
    check(run_bits(*c.gate, c.up, c.tokens, shared, s, x[i], y[i]) == expected[i],
          std::string(c.name) + ": shared eager output differs from its fp64-checked fresh workspace");
  }
  // A nonzero cap must still use the general route at each optimized bit width.
  set_cap("36");
  for (int i = 0; i < 3; ++i) {
    const Call &c = calls[i];
    const auto capped = check_swiglu(std::string(c.name) + " cap 36", *c.gate, *c.up, c.tokens, shared, 101000 + i);
    // Both 5080 routes use one K split at this shape: the output partition keeps its fold order.
    if (sms == 84) check(capped == expected[i], std::string(c.name) + ": differs from the unsplit capped route");
  }
  set_cap("0");
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  MOE_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < count; ++i) {
    const Call &c = calls[i];
    TL_RC(ignis_trellis_linear_swiglu(&c.gate->w, &c.up->w, x[i].p, c.tokens, y[i].p, out, &shared.ws, s));
  }
  MOE_CUDA(cudaStreamEndCapture(s, &graph));
  MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  for (int rep = 0; rep < 5; ++rep) {
    for (DeviceBytes &d : y) MOE_CUDA(cudaMemsetAsync(d.p, 0xff, d.bytes, s));
    MOE_CUDA(cudaGraphLaunch(exec, s));
    MOE_CUDA(cudaStreamSynchronize(s));
    for (int i = 0; i < count; ++i) {
      check(download<uint16_t>(y[i].p, expected[i].size()) == expected[i],
            std::string(calls[i].name) + ": shared graph replay " + std::to_string(rep) + " differs");
    }
  }
  MOE_CUDA(cudaGraphExecDestroy(exec));
  MOE_CUDA(cudaGraphDestroy(graph));
  MOE_CUDA(cudaStreamDestroy(s));
  std::printf("  T=8 MLP: same-K fp64, fallbacks, %d shared-workspace calls and 5 graph replays checked\n", count);
}

void determinism_arm() {
  DeviceWeight down, g, u;
  make_weight(down, 6, 17408, 5120, 70000);
  make_weight(g, 6, 5120, 17408, 70100);
  make_weight(u, 4, 5120, 17408, 70200);
  cudaStream_t s;
  MOE_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  for (const int tokens : {1, 8, 64, 300}) {
    for (int which = 0; which < 2; ++which) {
      const DeviceWeight &a = which == 0 ? down : g;
      const DeviceWeight *b = which == 0 ? nullptr : &u;
      const std::string name = std::string(which == 0 ? "down" : "swiglu") + " at " + std::to_string(tokens) + " tokens";
      Workspace ws(ignis_trellis_workspace_bytes(a.in, a.out, which == 0 ? 1 : 2, tokens));
      const std::vector<uint16_t> x = make_x(71000 + tokens, tokens, static_cast<int>(a.in));
      DeviceBytes d_x(x.size() * 2), d_y(static_cast<size_t>(tokens) * a.out * 2);
      upload(d_x, x);
      const auto first = run_bits(a, b, tokens, ws, s, d_x, d_y);
      check(run_bits(a, b, tokens, ws, s, d_x, d_y) == first, name + ": a second run differs");
      // Capped, the decode route splits for 36 SMs' CTAs (its bits may differ from the uncapped
      // run's, never between two capped runs); the prefill route does not split, so its bits stay.
      set_cap("36");
      check(ignis_trellis_sm_cap() == 36, "the cap reads back");
      const auto capped = run_bits(a, b, tokens, ws, s, d_x, d_y);
      check(run_bits(a, b, tokens, ws, s, d_x, d_y) == capped, name + ": a second run capped at 36 SMs differs");
      if (tokens > IGNIS_TRELLIS_DECODE_MAX_TOKENS) check(capped == first, name + ": the prefill capped at 36 SMs differs");
      set_cap("0");
      // Graph capture in global mode: an allocation inside the call would end the capture.
      MOE_CUDA(cudaMemsetAsync(d_y.p, 0, d_y.bytes, s));
      MOE_CUDA(cudaStreamSynchronize(s));
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      MOE_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      if (b == nullptr) {
        TL_RC(ignis_trellis_linear(&a.w, d_x.p, tokens, d_y.p, a.out, IGNIS_TRELLIS_STORE, &ws.ws, s));
      } else {
        TL_RC(ignis_trellis_linear_swiglu(&a.w, &b->w, d_x.p, tokens, d_y.p, a.out, &ws.ws, s));
      }
      MOE_CUDA(cudaStreamEndCapture(s, &graph));
      MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
      for (int rep = 0; rep < 3; ++rep) {
        MOE_CUDA(cudaMemsetAsync(d_y.p, 0, d_y.bytes, s));
        MOE_CUDA(cudaGraphLaunch(exec, s));
        MOE_CUDA(cudaStreamSynchronize(s));
        check(download<uint16_t>(d_y.p, first.size()) == first, name + ": graph replay " + std::to_string(rep) + " differs");
      }
      MOE_CUDA(cudaGraphExecDestroy(exec));
      MOE_CUDA(cudaGraphDestroy(graph));
      std::printf("  determinism %-26s: reruns (uncapped, cap 36) and 3 graph replays bit-identical\n", name.c_str());
    }
  }
  MOE_CUDA(cudaStreamDestroy(s));
}

// GitHub #339: one workspace serves a load's calls of every shape in turn, as the 27B's layer
// programs make them on one stream -- each call's bits are the ones it gives on a fresh
// workspace. A call's rotated activations follow its arrival counters; when those counters'
// room followed the call's own output blocks, a narrow call's activations sat where a wider
// call's counters are and left them nonzero (the down projection before the next layer's
// in_proj_qkv: 16 of its 80 output blocks reduced early).
void shared_workspace_arm() {
  DeviceWeight down, qkv, q, g, u;
  make_weight(down, 4, 17408, 5120, 90000);
  make_weight(qkv, 8, 5120, 10240, 90100);
  make_weight(q, 6, 5120, 12288, 90200);
  make_weight(g, 6, 5120, 17408, 90300);
  make_weight(u, 6, 5120, 17408, 90400);
  struct Call {
    const char *name;
    const DeviceWeight *a, *b;
    int tokens;
  };
  const Call calls[] = {{"down", &down, nullptr, 1},       {"in_proj_qkv", &qkv, nullptr, 1}, {"swiglu", &g, &u, 1},
                        {"down", &down, nullptr, 8},       {"q_proj", &q, nullptr, 1},        {"q_proj", &q, nullptr, 80},
                        {"in_proj_qkv", &qkv, nullptr, 2}, {"swiglu", &g, &u, 80},            {"in_proj_qkv", &qkv, nullptr, 1}};
  uint64_t bytes = 0;
  for (const Call &c : calls) {
    bytes = std::max(bytes, ignis_trellis_workspace_bytes(c.a->in, c.a->out, c.b == nullptr ? 1 : 2, c.tokens));
  }
  Workspace shared(bytes);
  cudaStream_t s;
  MOE_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const int failed_before = moe_test::g_failed;
  int i = 0;
  for (const Call &c : calls) {
    const std::vector<uint16_t> x = make_x(91000 + i, c.tokens, static_cast<int>(c.a->in));
    DeviceBytes d_x(x.size() * 2), d_y(static_cast<size_t>(c.tokens) * c.a->out * 2);
    upload(d_x, x);
    const auto in_turn = run_bits(*c.a, c.b, c.tokens, shared, s, d_x, d_y);
    Workspace fresh(ignis_trellis_workspace_bytes(c.a->in, c.a->out, c.b == nullptr ? 1 : 2, c.tokens));
    const auto alone = run_bits(*c.a, c.b, c.tokens, fresh, s, d_x, d_y);
    std::size_t differ = 0;
    for (std::size_t k = 0; k < alone.size(); ++k) differ += in_turn[k] != alone[k];
    check(differ == 0, std::string("shared workspace: call ") + std::to_string(i) + " (" + c.name + " at " +
                           std::to_string(c.tokens) + " tokens): " + std::to_string(differ) +
                           " outputs differ from a fresh workspace's");
    ++i;
  }
  MOE_CUDA(cudaStreamDestroy(s));
  if (moe_test::g_failed == failed_before) {
    std::printf("  shared workspace: %d calls of five shapes in turn, each bit-identical to its own fresh workspace\n", i);
  }
}

void rejects_arm() {
  DeviceWeight d;
  make_weight(d, 6, 256, 128, 80000);
  Workspace ws(ignis_trellis_workspace_bytes(256, 128, 1, 8));
  DeviceBytes x(8 * 256 * 2), y(8 * 128 * 2);
  check(ignis_trellis_linear(&d.w, x.p, 8, y.p, 128, 9, &ws.ws, nullptr) != 0, "an unknown epilogue is refused");
  check(ignis_trellis_linear(&d.w, x.p, 0, y.p, 128, 0, &ws.ws, nullptr) != 0, "zero tokens are refused");
  check(ignis_trellis_linear(&d.w, x.p, 8, y.p, 126, 0, &ws.ws, nullptr) != 0, "ldy below out is refused");
  ignis_trellis_workspace small = ws.ws;
  small.bytes = 1024;
  check(ignis_trellis_linear(&d.w, x.p, 8, y.p, 128, 0, &small, nullptr) != 0, "a short workspace is refused");
  check(ignis_trellis_linear(&d.w, static_cast<char *>(x.p) + 2, 8, y.p, 128, 0, &ws.ws, nullptr) != 0,
        "unaligned activations are refused");
  ignis_trellis_weight other = d.w;
  other.out = 256;
  check(ignis_trellis_linear_swiglu(&d.w, &other, x.p, 8, y.p, 128, &ws.ws, nullptr) != 0, "gate and up of different shapes are refused");
  check(ignis_trellis_workspace_bytes(100, 128, 1, 1) == 0, "a workspace query at in = 100 answers 0");
  check(ignis_trellis_workspace_bytes(128, 128, 3, 1) == 0, "a workspace query for three weights answers 0");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("dense trellis linear vs fp64 (|y - Y| <= 2^-8 |Y| + 2^-9 s)\n");
  TL_RC(ignis_trellis_prepare());
  fixtures_arm();
  shapes_arm();
  epilogues_arm();
  fp16_cap_arm();
  mlp_t8_arm();
  determinism_arm();
  shared_workspace_arm();
  rejects_arm();
  std::printf("worst share of the bound: decode %.3f, prefill %.3f\n", g_worst_decode, g_worst_prefill);
  if (moe_test::g_failed != 0) {
    std::fprintf(stderr, "test_trellis_linear: %d failure(s)\n", moe_test::g_failed);
    return 1;
  }
  std::printf("test_trellis_linear: OK\n");
  return 0;
}
