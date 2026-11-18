// GitHub #268 (spec runtime/09): the hq-e8-2b verify attention kernel's
// throughput form against its reference form, bit for bit -- OURS, not
// vendored.
//
// ADR 0031 lets a vendored kernel measured as the bottleneck carry a recorded
// patch, on one condition: where the op is exact, the patched op computes the
// same bits as the reference on the same input, and a test runs both and
// compares. gqa_attention_small_t_tc_partial_bf16_kernel's hq route is exact
// (a deterministic decode and the same MMA order), so that is this test:
//
//   IgnisThroughput = false  the reference's route: two K/V tiles, q fragments
//                            in registers, each row group-decoded straight
//                            from the code planes (hq_decode_row_group);
//   IgnisThroughput = true   the engine's route at 27B's 48 query rows: q in
//                            shared memory with K and V taking turns in one
//                            tile, each role's code rows staged one phase
//                            ahead by cp.async, the throughput group decoder.
//
// Both run on identical inputs -- the same paged cache encoded from the
// committed real K/V rows under their true dither seeds, the same q, the same
// positions -- over the shapes the verify round uses and the ones it may: one
// and several lanes, widths 1..8, masked columns, a column offset, a permuted
// table-row map, short windows (key-split units) and long ones (tile splits),
// split capacities (grid.y) of 4, 23, 40 and 85,
// the residual window on and off with cleared ring slots, and the fused append
// (GqaAppendInput) as well as a pre-filled cache (GqaCachedInput). Every byte
// of the partials must match, and with the append so must every byte the
// kernel wrote into the cache.
//
// The owned HQ27 fixed-256/D256 reducer is also checked against the unchanged
// generic reducer on independent reference partials, aliasing gate/output,
// nonfinite values, underflow/subnormal denominators, and a 257-split boundary.
// No expected output comes from the candidate.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU, a
// kernel error or a missing fixture fails this test.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ops/kernel/gqa_attention_decode_hq.cuh"
#include "ignis_gqa_hq_reduce.cuh"


using namespace ninfer;
using namespace ninfer::ops;

#ifndef IGNIS_HQ_KV_FIXTURE_PATH
#error "IGNIS_HQ_KV_FIXTURE_PATH must be defined by kernel/tests/CMakeLists.txt"
#endif

namespace {

int g_failed = 0;

void check(bool ok, const std::string& msg) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", msg.c_str());
        ++g_failed;
    }
}

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        const cudaError_t _err = (expr);                                                        \
        if (_err != cudaSuccess) {                                                              \
            std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));    \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                       \
    } while (0)

using Geometry = Gqa27Geometry;
constexpr int kTokenTile = 8;
constexpr int kSplits    = Geometry::DecodeSplits;
constexpr int kHeads     = Geometry::KVHeads;
constexpr int kRingWords = static_cast<int>(kGqaHqRecentKeys) / 32;

// The fixture's row payload. Its full validation (magic, version, geometry,
// checksum) is test_hq_codec_kv_rows.cu's; here a wrong file still fails.
struct Rows {
    int kv_heads = 0, layers = 0, per_block = 0;
    std::vector<std::uint16_t> bits;
};

Rows load_rows(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "FATAL: cannot open fixture %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    std::vector<std::uint8_t> data;
    std::fseek(f, 0, SEEK_END);
    data.resize(static_cast<std::size_t>(std::ftell(f)));
    std::fseek(f, 0, SEEK_SET);
    const bool read_ok = std::fread(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    auto u32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, data.data() + off, 4);
        return static_cast<int>(v);
    };
    if (!read_ok || data.size() < 32 || std::memcmp(data.data(), "IGNHQKV1", 8) != 0 ||
        u32(12) != kHqHeadDim || u32(16) != kHeads) {
        std::fprintf(stderr, "FATAL: %s is not the 27B IGNHQKV1 fixture\n", path);
        std::exit(EXIT_FAILURE);
    }
    Rows r;
    r.kv_heads = u32(16);
    r.layers   = u32(24);
    std::size_t off = 28 + 4 * static_cast<std::size_t>(r.layers);
    r.per_block     = u32(off + 4);
    const int total = u32(off + 8);
    off += 20;
    if (data.size() != off + static_cast<std::size_t>(total) * kHqHeadDim * 2) {
        std::fprintf(stderr, "FATAL: %s has a truncated payload\n", path);
        std::exit(EXIT_FAILURE);
    }
    r.bits.resize(static_cast<std::size_t>(total) * kHqHeadDim);
    std::memcpy(r.bits.data(), data.data() + off, r.bits.size() * 2);
    return r;
}

// One warp encodes one history row (table row t, key, head, role) from the
// fixture row of the same head and role, and dual-writes it into the side
// planes where the engine would (sink keys, and the ring slot of keys in the
// recent window).
__global__ void fill_history_kernel(const __nv_bfloat16* rows, int layers, int per_block,
                                    const std::int32_t* block_tables, int table_stride,
                                    const int* windows, int table_count, int max_window,
                                    GqaTcKVHq kv) {
    extern __shared__ float smem[];
    __shared__ std::int8_t signs[kHqHeadDim];
    hq_engine_signs_fill(signs);
    __syncthreads();
    const long long unit  = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const long long units = static_cast<long long>(table_count) * max_window * kHeads * 2;
    if (unit >= units) { return; }
    const bool role_v  = (unit & 1) != 0;
    const int head     = static_cast<int>((unit >> 1) % kHeads);
    const long long tk = (unit >> 1) / kHeads;
    const int key      = static_cast<int>(tk % max_window);
    const int t        = static_cast<int>(tk / max_window);
    if (key >= windows[t]) { return; }
    const int layer = (key / per_block + t) % layers;
    const int src   = ((layer * 2 + (role_v ? 1 : 0)) * kHeads + head) * per_block +
                    (key * 7 + 13 * t) % per_block;
    float* u            = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    const std::int32_t* table = block_tables + static_cast<long long>(t) * table_stride;
    hq_encode_row_warp(rows + static_cast<long long>(src) * kHqHeadDim, signs, 0, u, syms,
                       hq_row_codes_mut<Geometry>(role_v ? kv.codes_v : kv.codes_k, table, head, key),
                       hq_row_meta_mut<Geometry>(role_v ? kv.meta_v : kv.meta_k, table, head, key),
                       hq_dither_row_seed(head, key, role_v));
    if (kv.residual_k != nullptr &&
        (key < static_cast<int>(kGqaHqSinkKeys) || key >= windows[t] - static_cast<int>(kGqaHqRecentKeys))) {
        hq_store_rotated_row_warp(rows + static_cast<long long>(src) * kHqHeadDim, signs,
                                  hq_residual_row<Geometry>(role_v ? kv.residual_v : kv.residual_k,
                                                            t, head, key));
    }
}

__global__ void fill_bf16_kernel(__nv_bfloat16* out, long long n, std::uint32_t salt, float amp) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9u ^ salt;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    out[i] = __float2bfloat16((static_cast<float>(x & 0xFFFFu) / 65535.0f - 0.5f) * amp);
}

struct Case {
    const char* name;
    std::vector<int> windows;  // per lane: keys visible to its last column
    int tokens;                // width of the call (1..8)
    int full_width;            // columns per lane in q / pos
    int column_begin;          // first column of the call
    std::vector<int> valid;    // per lane valid columns (masked), empty = unmasked
    bool permute_tables;       // lane b reads table row B-1-b
    bool residual;
    bool append;
    int splits = kSplits;      // the launch's split capacity (grid.y); serving uses 4..85
};

template <typename CacheInput, bool Throughput>
void launch(const Case& c, const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
            GqaTcKVHq kv, const std::int32_t* tables, const std::int32_t* valid,
            const std::int32_t* table_rows, int table_stride, __nv_bfloat16* acc, float* m,
            float* l) {
    const dim3 grid(kHeads, c.splits, static_cast<unsigned>(c.windows.size()));
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, kTokenTile, 4, true, true, CacheInput,
                                                 GqaTcKVHq, Throughput>
        <<<grid, kGqaHqDecodeThreads>>>(q, input, pos, kv, tables, valid, table_rows, table_stride,
                                        c.tokens, c.full_width, c.column_begin, 262144,
                                        1.0f / 16.0f, acc, m, l);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// The unchanged generic 256-thread/D64 reducer is the exact oracle for
// the independent HQ27 fixed 256-thread/D256 specialization.
void launch_reduce(bool candidate, const Case& c, const __nv_bfloat16* acc,
                   const float* m, const float* l, const std::int32_t* pos,
                   const std::int32_t* valid, const __nv_bfloat16* gate,
                   __nv_bfloat16* out, cudaStream_t stream = nullptr) {
    const int lanes = static_cast<int>(c.windows.size());
    const dim3 grid(Geometry::QHeads, 4, c.tokens * lanes);
    const dim3 candidate_grid(Geometry::QHeads, kGqaHeadDim / kIgnisGqaHqReduceDChunk,
                              c.tokens * lanes);
    const auto selected = [&]<bool Multi, bool Mask, bool Offset>() {
        if (candidate) {
            ignis_gqa_hq_reduce_kernel<Geometry, kIgnisGqaHqReduceDChunk, false, Multi, Mask, Offset>
                <<<candidate_grid, kIgnisGqaHqReduceThreads, 0, stream>>>(acc, m, l, pos, valid, gate, c.tokens, c.full_width,
                                          c.column_begin, lanes, c.splits, out);
        } else {
            gqa_attention_small_t_reduce_output_kernel<Geometry, 64, false, Multi, Mask, Offset>
                <<<grid, 256, 0, stream>>>(acc, m, l, pos, valid, gate, c.tokens, c.full_width,
                                          c.column_begin, lanes, c.splits, out);
        }
    };
    const int flags = (lanes > 1 ? 4 : 0) | (!c.valid.empty() ? 2 : 0) | (c.column_begin != 0 ? 1 : 0);
    switch (flags) {
    case 0: selected.template operator()<false, false, false>(); break;
    case 1: selected.template operator()<false, false, true>(); break;
    case 2: selected.template operator()<false, true, false>(); break;
    case 3: selected.template operator()<false, true, true>(); break;
    case 4: selected.template operator()<true, false, false>(); break;
    case 5: selected.template operator()<true, false, true>(); break;
    case 6: selected.template operator()<true, true, false>(); break;
    default: selected.template operator()<true, true, true>(); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
std::vector<std::uint8_t> to_host(const T* d, std::size_t bytes) {
    std::vector<std::uint8_t> h(bytes);
    CUDA_CHECK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    return h;
}

std::size_t differing(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { n += a[i] != b[i] ? 1 : 0; }
    return n;
}

void run_case(const Case& c, const Rows& rows, const __nv_bfloat16* d_rows) {
    const int lanes = static_cast<int>(c.windows.size());
    int max_window  = 0;
    for (int w : c.windows) { max_window = w > max_window ? w : max_window; }
    const int pages_per_lane = (max_window + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const long long plane_rows = static_cast<long long>(pages_per_lane) * lanes * kHeads * kPagedKVPageSize;

    // Table row t owns pages t*P.., listed in reverse; lane b uses table row
    // b (or B-1-b), and its history/positions follow the table row it reads.
    std::vector<std::int32_t> tables(static_cast<std::size_t>(pages_per_lane) * lanes);
    for (int t = 0; t < lanes; ++t) {
        for (int p = 0; p < pages_per_lane; ++p) {
            tables[static_cast<std::size_t>(t) * pages_per_lane + p] = t * pages_per_lane + (pages_per_lane - 1 - p);
        }
    }
    std::vector<std::int32_t> table_rows(lanes);
    std::vector<int> table_windows(lanes);
    for (int b = 0; b < lanes; ++b) {
        table_rows[b]                = c.permute_tables ? lanes - 1 - b : b;
        table_windows[table_rows[b]] = c.windows[b];
    }
    // Lane b's columns end at its window: column j sits at position
    // window - full_width + j (all non-negative by construction below).
    std::vector<std::int32_t> pos(static_cast<std::size_t>(lanes) * c.full_width);
    for (int b = 0; b < lanes; ++b) {
        for (int j = 0; j < c.full_width; ++j) {
            pos[static_cast<std::size_t>(b) * c.full_width + j] =
                c.windows[b] - (c.column_begin + c.tokens) + j;
        }
    }
    // With the append, the call's own columns are written by the kernel, so
    // the history is filled only below them.
    std::vector<int> fill_windows(table_windows);
    if (c.append) {
        for (int b = 0; b < lanes; ++b) { fill_windows[table_rows[b]] = c.windows[b] - c.tokens; }
    }

    std::int32_t *d_tables, *d_table_rows, *d_pos, *d_valid = nullptr;
    int* d_windows;
    CUDA_CHECK(cudaMalloc(&d_tables, tables.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_table_rows, table_rows.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, pos.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_windows, lanes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tables, tables.data(), tables.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table_rows, table_rows.data(), table_rows.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_windows, fill_windows.data(), lanes * sizeof(int), cudaMemcpyHostToDevice));
    if (!c.valid.empty()) {
        CUDA_CHECK(cudaMalloc(&d_valid, lanes * 4));
        CUDA_CHECK(cudaMemcpy(d_valid, c.valid.data(), lanes * 4, cudaMemcpyHostToDevice));
    }

    GqaTcKVHq kv{};
    const std::size_t code_bytes = static_cast<std::size_t>(plane_rows) * kHqCodePlaneExtent;
    const std::size_t meta_bytes = static_cast<std::size_t>(plane_rows) * kHqMetaPlaneExtent;
    const std::size_t side_bytes = static_cast<std::size_t>(lanes) *
                                   (kGqaHqSinkKeys + kGqaHqRecentKeys) * kHeads * kHqHeadDim * 2;
    CUDA_CHECK(cudaMalloc(&kv.codes_k, code_bytes));
    CUDA_CHECK(cudaMalloc(&kv.codes_v, code_bytes));
    CUDA_CHECK(cudaMalloc(&kv.meta_k, meta_bytes));
    CUDA_CHECK(cudaMalloc(&kv.meta_v, meta_bytes));
    CUDA_CHECK(cudaMemset(kv.codes_k, 0, code_bytes));
    CUDA_CHECK(cudaMemset(kv.codes_v, 0, code_bytes));
    CUDA_CHECK(cudaMemset(kv.meta_k, 0, meta_bytes));
    CUDA_CHECK(cudaMemset(kv.meta_v, 0, meta_bytes));
    if (c.residual) {
        CUDA_CHECK(cudaMalloc(&kv.residual_k, side_bytes));
        CUDA_CHECK(cudaMalloc(&kv.residual_v, side_bytes));
        CUDA_CHECK(cudaMemset(kv.residual_k, 0, side_bytes));
        CUDA_CHECK(cudaMemset(kv.residual_v, 0, side_bytes));
        // Every ring slot valid but a few per table row, which fall back to
        // the codec rows (a rollback's cleared slots).
        std::vector<std::uint32_t> ring(static_cast<std::size_t>(lanes) * kRingWords, 0xFFFFFFFFu);
        for (int t = 0; t < lanes; ++t) {
            ring[static_cast<std::size_t>(t) * kRingWords + (t % kRingWords)] = 0xF0F0F0F0u;
            ring[static_cast<std::size_t>(t) * kRingWords + ((t + 7) % kRingWords)] &= ~0x1u;
        }
        CUDA_CHECK(cudaMalloc(&kv.ring_valid, ring.size() * 4));
        CUDA_CHECK(cudaMemcpy(kv.ring_valid, ring.data(), ring.size() * 4, cudaMemcpyHostToDevice));
    }
    {
        constexpr int kWarps = 8;
        const long long units = static_cast<long long>(lanes) * max_window * kHeads * 2;
        const std::size_t smem = kWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * 4;
        fill_history_kernel<<<static_cast<unsigned>((units + kWarps - 1) / kWarps), kWarps * 32, smem>>>(
            d_rows, rows.layers, rows.per_block, d_tables, pages_per_lane, d_windows, lanes, max_window, kv);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const long long q_elems = static_cast<long long>(lanes) * c.full_width * Geometry::QHeads * kGqaHeadDim;
    const long long kv_elems = static_cast<long long>(lanes) * c.full_width * kHeads * kGqaHeadDim;
    __nv_bfloat16 *d_q, *d_k_new, *d_v_new;
    CUDA_CHECK(cudaMalloc(&d_q, q_elems * 2));
    CUDA_CHECK(cudaMalloc(&d_k_new, kv_elems * 2));
    CUDA_CHECK(cudaMalloc(&d_v_new, kv_elems * 2));
    fill_bf16_kernel<<<static_cast<unsigned>((q_elems + 255) / 256), 256>>>(d_q, q_elems, 0x85EBCA6Bu, 4.0f);
    fill_bf16_kernel<<<static_cast<unsigned>((kv_elems + 255) / 256), 256>>>(d_k_new, kv_elems, 0xC2B2AE35u, 8.0f);
    fill_bf16_kernel<<<static_cast<unsigned>((kv_elems + 255) / 256), 256>>>(d_v_new, kv_elems, 0x27D4EB2Fu, 2.0f);
    CUDA_CHECK(cudaGetLastError());

    // The partials hold one block of every split for every lane; untouched
    // bytes keep a sentinel, so a split one route writes and the other skips
    // shows up as a difference.
    const std::size_t acc_bytes  = static_cast<std::size_t>(lanes) * c.splits * c.tokens * Geometry::QHeads * kGqaHeadDim * 2;
    const std::size_t stat_bytes = static_cast<std::size_t>(lanes) * c.splits * c.tokens * Geometry::QHeads * 4;
    __nv_bfloat16* d_acc;
    float *d_m, *d_l;
    CUDA_CHECK(cudaMalloc(&d_acc, acc_bytes));
    CUDA_CHECK(cudaMalloc(&d_m, stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_l, stat_bytes));

    // Snapshot the cache so both routes start from the same bytes.
    const auto codes_k0 = to_host(kv.codes_k, code_bytes), codes_v0 = to_host(kv.codes_v, code_bytes);
    const auto meta_k0 = to_host(kv.meta_k, meta_bytes), meta_v0 = to_host(kv.meta_v, meta_bytes);
    std::vector<std::uint8_t> side_k0, side_v0;
    const auto ring0 = c.residual ? to_host(kv.ring_valid, lanes * kRingWords * 4) : std::vector<std::uint8_t>{};
    if (c.residual) { side_k0 = to_host(kv.residual_k, side_bytes); side_v0 = to_host(kv.residual_v, side_bytes); }
    auto restore = [&]() {
        CUDA_CHECK(cudaMemcpy(kv.codes_k, codes_k0.data(), code_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kv.codes_v, codes_v0.data(), code_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kv.meta_k, meta_k0.data(), meta_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kv.meta_v, meta_v0.data(), meta_bytes, cudaMemcpyHostToDevice));
        if (c.residual) {
            CUDA_CHECK(cudaMemcpy(kv.residual_k, side_k0.data(), side_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(kv.residual_v, side_v0.data(), side_bytes, cudaMemcpyHostToDevice));
        }
        if (c.residual) CUDA_CHECK(cudaMemcpy(kv.ring_valid, ring0.data(), ring0.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_acc, 0xAB, acc_bytes));
        CUDA_CHECK(cudaMemset(d_m, 0xAB, stat_bytes));
        CUDA_CHECK(cudaMemset(d_l, 0xAB, stat_bytes));
    };
    struct Out {
        std::vector<std::uint8_t> acc, m, l, codes_k, codes_v, meta_k, meta_v, side_k, side_v, ring;
    };
    auto capture = [&]() {
        Out o;
        o.acc = to_host(d_acc, acc_bytes);
        o.m   = to_host(d_m, stat_bytes);
        o.l   = to_host(d_l, stat_bytes);
        o.codes_k = to_host(kv.codes_k, code_bytes);
        o.codes_v = to_host(kv.codes_v, code_bytes);
        o.meta_k  = to_host(kv.meta_k, meta_bytes);
        o.meta_v  = to_host(kv.meta_v, meta_bytes);
        if (c.residual) { o.side_k = to_host(kv.residual_k, side_bytes); o.side_v = to_host(kv.residual_v, side_bytes); }
        if (c.residual) o.ring = to_host(kv.ring_valid, ring0.size());
        return o;
    };
    auto run = [&](bool throughput) {
        restore();
        if (c.append) {
            const GqaAppendInput input{d_k_new, d_v_new};
            if (throughput) {
                launch<GqaAppendInput, true>(c, d_q, input, d_pos, kv, d_tables, d_valid, d_table_rows, pages_per_lane, d_acc, d_m, d_l);
            } else {
                launch<GqaAppendInput, false>(c, d_q, input, d_pos, kv, d_tables, d_valid, d_table_rows, pages_per_lane, d_acc, d_m, d_l);
            }
        } else if (throughput) {
            launch<GqaCachedInput, true>(c, d_q, GqaCachedInput{}, d_pos, kv, d_tables, d_valid, d_table_rows, pages_per_lane, d_acc, d_m, d_l);
        } else {
            launch<GqaCachedInput, false>(c, d_q, GqaCachedInput{}, d_pos, kv, d_tables, d_valid, d_table_rows, pages_per_lane, d_acc, d_m, d_l);
        }
        return capture();
    };
    const Out ref  = run(false);
    const Out fast = run(true);

    // The reference must have produced real attention, not only neutral
    // partials: some split of some row has l > 0.
    bool any_mass = false;
    for (std::size_t i = 0; i + 4 <= ref.l.size() && !any_mass; i += 4) {
        float v;
        std::memcpy(&v, ref.l.data() + i, 4);
        any_mass = v > 0.0f && v < 1e30f;
    }
    const std::size_t d_acc_n = differing(ref.acc, fast.acc), d_m_n = differing(ref.m, fast.m),
                      d_l_n = differing(ref.l, fast.l);
    const std::size_t d_cache = differing(ref.codes_k, fast.codes_k) + differing(ref.codes_v, fast.codes_v) +
                                differing(ref.meta_k, fast.meta_k) + differing(ref.meta_v, fast.meta_v) +
                                differing(ref.side_k, fast.side_k) + differing(ref.side_v, fast.side_v) + differing(ref.ring, fast.ring);
    const std::size_t appended = differing(codes_k0, ref.codes_k) + differing(codes_v0, ref.codes_v);
    std::printf("%-44s partial bytes differing: acc %zu of %zu, m %zu, l %zu; cache bytes differing %zu "
                "(append changed %zu code bytes)\n",
                c.name, d_acc_n, ref.acc.size(), d_m_n, d_l_n, d_cache, appended);
    check(any_mass, std::string(c.name) + ": the reference route produced no attention mass");
    check(d_acc_n == 0 && d_m_n == 0 && d_l_n == 0,
          std::string(c.name) + ": the throughput route's partials differ from the reference route's");
    check(d_cache == 0, std::string(c.name) + ": the routes left different cache bytes");
    check(!c.append || appended > 0, std::string(c.name) + ": the fused append wrote nothing");

    const auto float_bits = [](std::uint32_t bits) { float value; std::memcpy(&value, &bits, 4); return value; };
    // The independent original partial reference is the fixed reducer input;
    // never derive expected final output from the cached-weight candidate.
    __nv_bfloat16 *d_gate, *d_reduce_original, *d_reduce_candidate;
    const std::size_t output_bytes = static_cast<std::size_t>(q_elems) * 2;
    CUDA_CHECK(cudaMalloc(&d_gate, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_reduce_original, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_reduce_candidate, output_bytes));
    fill_bf16_kernel<<<static_cast<unsigned>((q_elems + 255) / 256), 256>>>(d_gate, q_elems, 0xA341316Cu, 8.0f);
    CUDA_CHECK(cudaGetLastError());
    const auto gate0 = to_host(d_gate, output_bytes);
    const auto upload_partials = [&](const Out& x) {
        CUDA_CHECK(cudaMemcpy(d_acc, x.acc.data(), acc_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, x.m.data(), stat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_l, x.l.data(), stat_bytes, cudaMemcpyHostToDevice));
    };
    const auto reduce_gate = [&](const char* variant, const Out& inputs, bool alias,
                                 const std::vector<std::uint8_t>& gate_bits) {
        upload_partials(inputs);
        CUDA_CHECK(cudaMemcpy(d_gate, gate_bits.data(), output_bytes, cudaMemcpyHostToDevice));
        if (alias) {
            CUDA_CHECK(cudaMemcpy(d_reduce_original, gate_bits.data(), output_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_reduce_candidate, gate_bits.data(), output_bytes, cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemset(d_reduce_original, 0xA5, output_bytes));
            CUDA_CHECK(cudaMemset(d_reduce_candidate, 0xA5, output_bytes));
        }
        launch_reduce(false, c, d_acc, d_m, d_l, d_pos, d_valid,
            alias ? d_reduce_original : d_gate, d_reduce_original);
        const auto expected = to_host(d_reduce_original, output_bytes);
        launch_reduce(true, c, d_acc, d_m, d_l, d_pos, d_valid,
            alias ? d_reduce_candidate : d_gate, d_reduce_candidate);
        const auto actual = to_host(d_reduce_candidate, output_bytes);
        const std::size_t differences = differing(expected, actual);
        std::printf("REDUCE_GATE %s variant %s alias %d output_bytes %zu differing_bytes %zu\n",
            c.name, variant, alias ? 1 : 0, output_bytes, differences);
        check(differences == 0, std::string(c.name) + ": reducer output mismatch");
        return expected;
    };
    reduce_gate("real-original-partials", ref, false, gate0);
    reduce_gate("real-original-partials", ref, true, gate0);
    {
        std::vector<std::uint8_t> special_gate = gate0;
        std::vector<std::uint16_t> gb(special_gate.size() / 2);
        std::memcpy(gb.data(), special_gate.data(), special_gate.size());
        for (std::size_t i = 0; i < special_gate.size() / 2; ++i) {
            if (i % 37 == 0) gb[i] = 0x7FC1u;
            else if (i % 41 == 0) gb[i] = 0x7F80u;
            else if (i % 43 == 0) gb[i] = 0xFF80u;
        }
        std::memcpy(special_gate.data(), gb.data(), special_gate.size());
        if (c.splits > Geometry::DecodeSplits) {
            auto negative = pos;
            for (int b = 0; b < lanes; ++b) negative[b * c.full_width + c.column_begin + c.tokens - 1] = -1;
            CUDA_CHECK(cudaMemcpy(d_pos, negative.data(), negative.size() * 4, cudaMemcpyHostToDevice));
        }
        for (int variant = 0; variant < 4; ++variant) {
            Out edge = ref;
            std::vector<float> em(edge.m.size() / 4), el(edge.l.size() / 4);
            std::vector<std::uint16_t> ea(edge.acc.size() / 2);
            std::memcpy(em.data(), edge.m.data(), edge.m.size());
            std::memcpy(el.data(), edge.l.data(), edge.l.size());
            std::memcpy(ea.data(), edge.acc.data(), edge.acc.size());
            for (std::size_t i = 0; i < edge.m.size() / 4; ++i) {
                em[i] = variant == 0 ? float_bits(0xFF800000u) : -static_cast<float>((i * 17) % 201);
                el[i] = variant == 0 ? 0.0f : 0.25f * static_cast<float>(1 + i % 7);
                if (variant == 3) {
                    const std::size_t split = (i / (Geometry::QHeads * c.tokens)) % c.splits;
                    em[i] = split == 0 ? 0.0f : -102.0f;
                    el[i] = split == 0 ? 0.0f : 1.0f;
                }
                if (variant == 2) {
                    if (i % 17 == 0) el[i] = float_bits(0x7FC12345u);
                    else if (i % 13 == 0) el[i] = -1.0f;
                    else if (i % 11 == 0) el[i] = -0.0f;
                    else if (i % 7 == 0) el[i] = 0.0f;
                    if (i % 19 == 0) em[i] = float_bits(0x7FC54321u);
                }
            }
            for (std::size_t i = 0; i < edge.acc.size() / 2; ++i) {
                if (variant == 3) ea[i] = 0x3F80u;
                else if (variant == 0 || i % 101 == 0) ea[i] = 0x7FC1u;
                else if (i % 103 == 0) ea[i] = 0x7F80u;
                else if (i % 107 == 0) ea[i] = 0xFF80u;
                else ea[i] = static_cast<std::uint16_t>(0x3F00u + (i % 128));
            }
            const char* name = variant == 0 ? "all-neutral" : variant == 1 ? "underflow-inf-nan-acc" : variant == 2 ? "mixed-l-m-nonfinite" : "subnormal-denominator";
            std::memcpy(edge.m.data(), em.data(), edge.m.size());
            std::memcpy(edge.l.data(), el.data(), edge.l.size());
            std::memcpy(edge.acc.data(), ea.data(), edge.acc.size());
            reduce_gate(name, edge, false, special_gate);
            reduce_gate(name, edge, true, special_gate);
        }
        CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaFree(d_gate)); CUDA_CHECK(cudaFree(d_reduce_original)); CUDA_CHECK(cudaFree(d_reduce_candidate));

    CUDA_CHECK(cudaFree(d_tables));
    CUDA_CHECK(cudaFree(d_table_rows));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_windows));
    if (d_valid != nullptr) { CUDA_CHECK(cudaFree(d_valid)); }
    CUDA_CHECK(cudaFree(kv.codes_k));
    CUDA_CHECK(cudaFree(kv.codes_v));
    CUDA_CHECK(cudaFree(kv.meta_k));
    CUDA_CHECK(cudaFree(kv.meta_v));
    if (c.residual) {
        CUDA_CHECK(cudaFree(kv.residual_k));
        CUDA_CHECK(cudaFree(kv.residual_v));
        CUDA_CHECK(cudaFree(kv.ring_valid));
    }
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k_new));
    CUDA_CHECK(cudaFree(d_v_new));
    CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_l));
}

} // namespace

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    const Rows rows = load_rows(IGNIS_HQ_KV_FIXTURE_PATH);
    __nv_bfloat16* d_rows;
    CUDA_CHECK(cudaMalloc(&d_rows, rows.bits.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_rows, rows.bits.data(), rows.bits.size() * 2, cudaMemcpyHostToDevice));

    const std::vector<Case> cases = {
        {"1 lane, 40 keys, width 8", {40}, 8, 8, 0, {}, false, false, false},
        {"1 lane, 700 keys, width 5, residual", {700}, 5, 5, 0, {}, false, true, false},
        {"1 lane, 5000 keys, width 1, append, residual", {5000}, 1, 1, 0, {}, false, true, true},
        {"1 lane, 70000 keys, width 8, append, residual", {70000}, 8, 8, 0, {}, false, true, true},
        {"2 lanes, 20000/17000 keys, width 8", {20000, 17000}, 8, 8, 0, {}, false, false, false},
        {"3 lanes, mixed, masked, permuted, append", {5000, 1200, 33}, 8, 8, 0, {8, 6, 3}, true, true, true},
        {"2 lanes, offset columns 4..11 of 12", {9000, 600}, 8, 12, 4, {}, false, true, false},
        {"8 lanes, 3000 keys, width 8, masked, append", {3000, 3100, 2900, 3050, 3000, 2990, 3010, 3070},
         8, 8, 0, {8, 8, 8, 7, 8, 8, 5, 8}, false, true, true},
        // Fewer splits than the 85 of the serving envelope's capacity: long
        // per-split tile loops, and the capacity clamping the active count.
        {"2 lanes, 20000/17000 keys, 4 splits, residual", {20000, 17000}, 8, 8, 0, {}, false, true, false, 4},
        {"3 lanes, mixed, masked, append, 23 splits", {5000, 1200, 33}, 8, 8, 0, {8, 6, 3}, true, true, true, 23},
        {"1 lane, 70000 keys, width 8, append, 40 splits", {70000}, 8, 8, 0, {}, false, true, true, 40},
    };
    for (const Case& c : cases) { run_case(c, rows, d_rows); }
    const std::vector<Case> additional = {
        {"append-C1-T8-5000", {5000}, 8, 8, 0, {}, false, true, true},
        {"append-C1-T8-19992", {19992}, 8, 8, 0, {}, false, true, true},
        {"append-C1-T8-70000", {70000}, 8, 8, 0, {}, false, true, true},
        {"cached-C1-T8-19992", {19992}, 8, 8, 0, {}, false, true, false},
        {"append-C1-T1-19992", {19992}, 1, 1, 0, {}, false, true, true},
        {"append-C1-T5-19992", {19992}, 5, 5, 0, {}, false, true, true},
        {"append-C2-T8-19992", {19992,19992}, 8, 8, 0, {}, false, true, true},
    };
    for (const Case& c : additional) run_case(c, rows, d_rows);
    run_case({"direct-reducer-257-splits", {8}, 1, 1, 0, {}, false, false, false, 257}, rows, d_rows);
    CUDA_CHECK(cudaFree(d_rows));

    if (g_failed != 0) {
        std::fprintf(stderr, "hq_verify_exact: %d failures\n", g_failed);
        return EXIT_FAILURE;
    }
    std::printf("hq_verify_exact: ALL PASSED\n");
    return 0;
}
