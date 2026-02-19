#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── NF4 lookup table (QLoRA, 16 quantile values of N(0,1)) ──────────────────
__constant__ float c_nf4[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
     0.0f,
     0.07958029955625534f,
     0.16093020141124725f,
     0.24611230194568634f,
     0.33791524171829224f,
     0.44070982933044434f,
     0.5626170039176941f,
     0.7229568362236023f,
     1.0f,
};

// ── NF4 Dequantization Kernel ────────────────────────────────────────────────
//
// Each thread processes ONE packed byte (= 2 elements).
//
// Two-level scale recovery:
//   scale = code2[absmax_q[block_idx]] * absmax2[block_idx / group_size] + offset
//
// Dequantized value:
//   output[i] = NF4_TABLE[4bit_index] * scale
//
// Vectorized store: two fp16 values packed into one uint32_t write.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void nf4_dequantize_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half*    __restrict__ absmax2,
    const half*    __restrict__ code2,
    float          offset,
    half*          __restrict__ output,
    int64_t        total_elements,
    int            blocksize,
    int            group_size)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t num_pairs = (total_elements + 1) / 2;
    if (tid >= num_pairs) return;

    int64_t elem0 = tid * 2;

    // Unpack two 4-bit NF4 indices from one byte
    uint8_t packed = packed_weights[tid];
    uint8_t idx_lo = (packed >> 4) & 0x0F;  // upper nibble -> even element
    uint8_t idx_hi = packed & 0x0F;         // lower nibble -> odd element

    // ── Recover block-level scale for element 0 ──
    int blk0  = static_cast<int>(elem0 / blocksize);
    int grp0  = blk0 / group_size;
    float sc0 = __half2float(code2[absmax_q[blk0]])
              * __half2float(absmax2[grp0])
              + offset;

    if (tid == 0) {
        printf("KERNEL tid=0: packed=%u lo=%u hi=%u blk=%d grp=%d aq=%u\n",
               packed, idx_lo, idx_hi, blk0, grp0, absmax_q[blk0]);
        printf("KERNEL tid=0: code2[aq]=%.6f absmax2[grp]=%.6f off=%.6f sc=%.6f\n",
               __half2float(code2[absmax_q[blk0]]), __half2float(absmax2[grp0]),
               offset, sc0);
        printf("KERNEL tid=0: nf4[lo]=%.6f nf4[hi]=%.6f val0=%.6f\n",
               c_nf4[idx_lo], c_nf4[idx_hi], c_nf4[idx_lo] * sc0);
    }

    half h0 = __float2half(c_nf4[idx_lo] * sc0);

    // ── Element 1 (with boundary guard) ──
    if (elem0 + 1 < total_elements) {
        int blk1 = static_cast<int>((elem0 + 1) / blocksize);
        float sc1 = sc0;
        if (blk1 != blk0) {
            int grp1 = blk1 / group_size;
            sc1 = __half2float(code2[absmax_q[blk1]])
                * __half2float(absmax2[grp1])
                + offset;
        }
        half h1 = __float2half(c_nf4[idx_hi] * sc1);

        // Vectorized 32-bit store: two fp16 packed into one uint32_t
        uint32_t packed_out = static_cast<uint32_t>(__half_as_ushort(h0))
                            | (static_cast<uint32_t>(__half_as_ushort(h1)) << 16);
        reinterpret_cast<uint32_t*>(output)[tid] = packed_out;
    } else {
        // Last element when total is odd — scalar store
        output[elem0] = h0;
    }
}

// ── CLI helpers ──────────────────────────────────────────────────────────────

static const char* get_arg(int argc, char** argv, const char* key,
                           const char* fallback = nullptr) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    return fallback;
}

static int get_int_arg(int argc, char** argv, const char* key, int fallback) {
    const char* v = get_arg(argc, argv, key);
    return v ? atoi(v) : fallback;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* input_path  = get_arg(argc, argv, "--input");
    const char* output_path = get_arg(argc, argv, "--output");
    int group_size = get_int_arg(argc, argv, "--group_size", 256);
    int warmup     = get_int_arg(argc, argv, "--warmup", 3);
    int iters      = get_int_arg(argc, argv, "--iters", 10);

    if (!input_path || !output_path) {
        fprintf(stderr,
                "Usage: %s --input BIN --output BIN "
                "[--compute_type fp16|bf16] [--group_size N] "
                "[--warmup N] [--iters N]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    // ── 1. Read binary input file ────────────────────────────────────────────
    FILE* fin = fopen(input_path, "rb");
    if (!fin) { perror("fopen(input)"); return EXIT_FAILURE; }

    int64_t num_rows, num_cols;
    int32_t blocksize;
    fread(&num_rows,  sizeof(int64_t), 1, fin);
    fread(&num_cols,  sizeof(int64_t), 1, fin);
    fread(&blocksize, sizeof(int32_t), 1, fin);

    int64_t total     = num_rows * num_cols;
    int     num_blocks = static_cast<int>((total + blocksize - 1) / blocksize);
    int64_t n_padded  = static_cast<int64_t>(num_blocks) * blocksize;
    int64_t num_packed = n_padded / 2;   // bitsandbytes pads to full blocks
    int     num_groups = (num_blocks + group_size - 1) / group_size;

    printf("Shape: %ldx%ld  blocksize=%d  group_size=%d\n",
           (long)num_rows, (long)num_cols, blocksize, group_size);
    printf("Elements=%ld  packed_bytes=%ld  blocks=%d  groups=%d\n",
           (long)total, (long)num_packed, num_blocks, num_groups);

    std::vector<uint8_t>  h_packed(num_packed);
    std::vector<uint8_t>  h_absmax_q(num_blocks);
    std::vector<uint16_t> h_absmax2(num_groups);
    std::vector<uint16_t> h_code2(256);
    float h_offset = 0.0f;

    fread(h_packed.data(),   1,              num_packed,  fin);
    fread(h_absmax_q.data(), 1,              num_blocks,  fin);
    fread(h_absmax2.data(),  sizeof(uint16_t), num_groups, fin);
    fread(h_code2.data(),    sizeof(uint16_t), 256,        fin);
    fread(&h_offset,         sizeof(float),    1,          fin);
    fclose(fin);

    printf("Offset: %.6f\n", h_offset);

    // Debug: inspect data read from file
    printf("DEBUG packed[0..3]: %u %u %u %u\n",
           h_packed[0], h_packed[1], h_packed[2], h_packed[3]);
    printf("DEBUG absmax_q[0..3]: %u %u %u %u\n",
           h_absmax_q[0], h_absmax_q[1], h_absmax_q[2], h_absmax_q[3]);
    printf("DEBUG absmax2[0]: 0x%04x\n", h_absmax2[0]);
    printf("DEBUG code2[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
           h_code2[0], h_code2[1], h_code2[2], h_code2[3]);

    // ── 2. Allocate device memory & copy ─────────────────────────────────────
    uint8_t* d_packed    = nullptr;
    uint8_t* d_absmax_q  = nullptr;
    half*    d_absmax2   = nullptr;
    half*    d_code2     = nullptr;
    half*    d_output    = nullptr;

    CHECK_CUDA(cudaMalloc(&d_packed,   num_packed));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, num_blocks));
    CHECK_CUDA(cudaMalloc(&d_absmax2,  num_groups * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_code2,    256 * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output,   total * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_packed,   h_packed.data(),   num_packed,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), num_blocks,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2,  h_absmax2.data(),  num_groups * sizeof(half),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2,    h_code2.data(),    256 * sizeof(half),
                           cudaMemcpyHostToDevice));

    // ── 3. Launch configuration ──────────────────────────────────────────────
    const int threads_per_block = 256;
    int64_t num_pairs = (total + 1) / 2;
    int grid_size = static_cast<int>((num_pairs + threads_per_block - 1)
                                     / threads_per_block);

    // ── 4. Warmup ────────────────────────────────────────────────────────────
    for (int i = 0; i < warmup; ++i) {
        nf4_dequantize_kernel<<<grid_size, threads_per_block>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
            d_output, total, blocksize, group_size);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // ── 5. Timed iterations ──────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));

    for (int i = 0; i < iters; ++i) {
        nf4_dequantize_kernel<<<grid_size, threads_per_block>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
            d_output, total, blocksize, group_size);
    }

    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    float avg_ms = elapsed_ms / iters;

    double bytes_in  = static_cast<double>(num_packed + num_blocks
                                           + num_groups * 2 + 256 * 2);
    double bytes_out = static_cast<double>(total) * 2;
    double bw_gbs    = (bytes_in + bytes_out) / (avg_ms * 1e6);

    printf("Kernel time : %.4f ms (avg of %d iters)\n", avg_ms, iters);
    printf("Bandwidth   : %.2f GB/s\n", bw_gbs);

    // ── 6. Copy back & write output ──────────────────────────────────────────
    std::vector<uint16_t> h_output(total);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, total * sizeof(half),
                           cudaMemcpyDeviceToHost));

    int nonzero = 0;
    for (int i = 0; i < 100 && i < total; ++i)
        if (h_output[i] != 0) nonzero++;
    printf("DEBUG: first 100 outputs, %d non-zero. raw[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
           nonzero, h_output[0], h_output[1], h_output[2], h_output[3]);

    FILE* fout = fopen(output_path, "wb");
    if (!fout) { perror("fopen(output)"); return EXIT_FAILURE; }
    fwrite(h_output.data(), sizeof(uint16_t), total, fout);
    fclose(fout);

    printf("Output: %ld fp16 values -> %s\n", (long)total, output_path);

    // ── 7. Cleanup ───────────────────────────────────────────────────────────
    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_absmax_q));
    CHECK_CUDA(cudaFree(d_absmax2));
    CHECK_CUDA(cudaFree(d_code2));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    return 0;
}
