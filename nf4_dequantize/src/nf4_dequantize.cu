#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cfloat>
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ── 主机端 half ↔ float 转换辅助函数 ─────────────────────────────────────
static inline float half_to_float_host(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── 你的 NF4 表（保留不变） ───────────────────────────────────────────────
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

// ── BNB 解量化函数声明（正确签名） ─────────────────────────────────────────
// 注意：BNB 使用单层量化格式，需要 float* absmax 和总元素数 n
extern "C" {
void cdequantize_blockwise_fp16_nf4(
    float* code,           // NF4 码本 (通常传 nullptr，BNB 内部有默认表)
    unsigned char* A,      // packed_weights
    float* absmax,         // 块级最大值 (float*, 单层量化格式)
    half* out,             // 输出
    int blocksize,         // 块大小
    const int n,           // 总元素数量
    cudaStream_t stream
);
}

// ── 将双重量化的 absmax 解码为单层量化格式 ─────────────────────────────────
// 双重量化: absmax_q[blk] 索引 code2 表，再乘以 absmax2[grp]
__global__ void decode_double_quant_absmax_kernel(
    const uint8_t* __restrict__ absmax_q,
    const half*    __restrict__ absmax2,
    const half*    __restrict__ code2,
    float          offset,
    float*         __restrict__ absmax_out,
    int            num_blocks,
    int            group_size)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    
    int grp = blk / group_size;
    float scale = __half2float(code2[absmax_q[blk]])
                * __half2float(absmax2[grp])
                + offset;
    absmax_out[blk] = scale;
}

// ── float → half 转换核（用于 BNB 输出转换） ─────────────────────────────
__global__ void float2half_kernel(float* in, half* out, int64_t total_elements) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        out[i] = __float2half(in[i]);
    }
}

// ── 自己实现的 NF4 解量化核  ────────────────────────────────────
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

    // Unpack two 4-bit NF4 indices
    // BNB packing convention (verified from BNB source code):
    //   upper nibble (bits 4-7) -> even element (elem0)
    //   lower nibble (bits 0-3) -> odd element (elem0 + 1)
    uint8_t packed = packed_weights[tid];
    uint8_t idx_even = (packed >> 4) & 0x0F;  // upper nibble -> even element (elem0)
    uint8_t idx_odd  = packed & 0x0F;         // lower nibble -> odd element (elem0 + 1)

    // ── Recover block-level scale for element 0  ──
    int blk0  = static_cast<int>(elem0 / blocksize);
    int grp0  = blk0 / group_size;
    float sc0 = __half2float(code2[absmax_q[blk0]])
              * __half2float(absmax2[grp0])
              + offset;

    half h0 = __float2half(c_nf4[idx_even] * sc0);

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
        half h1 = __float2half(c_nf4[idx_odd] * sc1);

        // Vectorized 32-bit store: two fp16 packed into one uint32_t
        uint32_t packed_out = static_cast<uint32_t>(__half_as_ushort(h0))
                            | (static_cast<uint32_t>(__half_as_ushort(h1)) << 16);
        reinterpret_cast<uint32_t*>(output)[tid] = packed_out;
    } else {
        // Last element when total is odd — scalar store
        output[elem0] = h0;
    }
}

// ── 调用 BNB 解量化（用于对比） ─────────────────────────────────────────
// 注意：需要预先分配缓冲区，避免在计时循环内分配/释放内存
void run_bnb_dequantize(
    const uint8_t* d_packed,
    const uint8_t* d_absmax_q,
    const half* d_absmax2,
    const half* d_code2,
    float offset,
    float* d_absmax_decoded,  // 预分配的 absmax 解码缓冲区 (num_blocks * sizeof(float))
    half* d_output,
    int64_t total_elems,
    int num_blocks,
    int blocksize,
    int group_size,
    cudaStream_t stream)
{
    // Step 1: 将双重量化的 absmax 解码为单层量化格式
    int threads = 256;
    int blocks_for_absmax = (num_blocks + threads - 1) / threads;
    decode_double_quant_absmax_kernel<<<blocks_for_absmax, threads, 0, stream>>>(
        d_absmax_q, d_absmax2, d_code2, offset,
        d_absmax_decoded, num_blocks, group_size
    );

    // Step 2: 调用 BNB 官方解量化函数
    // 注意：code 参数传 nullptr，BNB 内部使用默认 NF4 码本
    cdequantize_blockwise_fp16_nf4(
        nullptr,  // code (使用内部默认 NF4 表)
        const_cast<uint8_t*>(d_packed),
        d_absmax_decoded,
        d_output,
        blocksize,
        static_cast<int>(total_elems),
        stream
    );
}

// ── 精度对比函数（集成到 CUDA 代码中） ───────────────────────────────────
void compare_accuracy(
    half* d_my_output,
    half* d_bnb_output,
    int64_t total_elems)
{
    // 拷贝结果到主机
    std::vector<uint16_t> h_my_output(total_elems);
    std::vector<uint16_t> h_bnb_output(total_elems);
    
    CHECK_CUDA(cudaMemcpy(h_my_output.data(), d_my_output, total_elems * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_bnb_output.data(), d_bnb_output, total_elems * sizeof(half), cudaMemcpyDeviceToHost));

    // 计算误差
    double mae = 0.0;
    float max_diff = 0.0f;
    float min_ref = FLT_MAX, max_ref = -FLT_MAX;
    float min_our = FLT_MAX, max_our = -FLT_MAX;

    for (int64_t i = 0; i < total_elems; ++i) {
        float my_val = half_to_float_host(h_my_output[i]);
        float bnb_val = half_to_float_host(h_bnb_output[i]);
        
        // 更新 BNB 参考范围
        min_ref = fminf(min_ref, bnb_val);
        max_ref = fmaxf(max_ref, bnb_val);
        
        // 更新我们的输出范围
        min_our = fminf(min_our, my_val);
        max_our = fmaxf(max_our, my_val);
        
        // 计算误差
        float diff = fabsf(my_val - bnb_val);
        mae += diff;
        max_diff = fmaxf(max_diff, diff);
    }

    mae /= total_elems;
    float value_range = max_ref - min_ref;
    float relative_mae = (value_range > 1e-8f) ? static_cast<float>(mae / value_range) : 0.0f;

    // 打印精度结果
    printf("\n=== Accuracy Comparison (My Kernel vs BNB) ===\n");
    printf("  MAE (absolute):  %.6f\n", mae);
    printf("  MAE (relative):  %.6f\n", relative_mae);
    printf("  Max diff:        %.6f\n", max_diff);
    printf("  Reference range: [%.4f, %.4f]\n", min_ref, max_ref);
    printf("  Our range:       [%.4f, %.4f]\n", min_our, max_our);

    // 检查是否通过 MAE < 1e-2
    if (relative_mae < 1e-2f) {
        printf("  ✅ PASSED: relative MAE < 1e-2\n");
    } else {
        printf("  ❌ FAILED: relative MAE >= 1e-2\n");
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
    
    if (fread(&num_rows,  sizeof(int64_t), 1, fin) != 1 ||
        fread(&num_cols,  sizeof(int64_t), 1, fin) != 1 ||
        fread(&blocksize, sizeof(int32_t), 1, fin) != 1) {
        fprintf(stderr, "Error: Failed to read header from input file\n");
        fclose(fin);
        return EXIT_FAILURE;
    }

    // 输入参数验证
    if (num_rows <= 0 || num_cols <= 0) {
        fprintf(stderr, "Error: Invalid dimensions (rows=%ld, cols=%ld)\n",
                (long)num_rows, (long)num_cols);
        fclose(fin);
        return EXIT_FAILURE;
    }
    if (blocksize <= 0) {
        fprintf(stderr, "Error: Invalid blocksize (%d)\n", blocksize);
        fclose(fin);
        return EXIT_FAILURE;
    }
    if (group_size <= 0) {
        fprintf(stderr, "Error: Invalid group_size (%d)\n", group_size);
        fclose(fin);
        return EXIT_FAILURE;
    }

    int64_t total     = num_rows * num_cols;
    int     num_blocks = static_cast<int>((total + blocksize - 1) / blocksize);
    int64_t num_packed = (total + 1) / 2;
    int     num_groups = (num_blocks + group_size - 1) / group_size;

    printf("=== Input Info ===\n");
    printf("Shape: %ldx%ld  blocksize=%d  group_size=%d\n",
           (long)num_rows, (long)num_cols, blocksize, group_size);
    printf("Elements=%ld  packed_bytes=%ld  blocks=%d  groups=%d\n",
           (long)total, (long)num_packed, num_blocks, num_groups);

    std::vector<uint8_t>  h_packed(num_packed);
    std::vector<uint8_t>  h_absmax_q(num_blocks);
    std::vector<uint16_t> h_absmax2(num_groups);
    std::vector<uint16_t> h_code2(256);
    float h_offset = 0.0f;

    size_t r1 = fread(h_packed.data(),   1,               num_packed,  fin);
    size_t r2 = fread(h_absmax_q.data(), 1,               num_blocks,  fin);
    size_t r3 = fread(h_absmax2.data(),  sizeof(uint16_t), num_groups, fin);
    size_t r4 = fread(h_code2.data(),    sizeof(uint16_t), 256,        fin);
    size_t r5 = fread(&h_offset,         sizeof(float),    1,          fin);
    fclose(fin);

    if (r1 != static_cast<size_t>(num_packed) || 
        r2 != static_cast<size_t>(num_blocks) ||
        r3 != static_cast<size_t>(num_groups) || 
        r4 != 256 || r5 != 1) {
        fprintf(stderr, "Error: Input file is truncated or corrupted\n");
        fprintf(stderr, "  Expected: packed=%ld, absmax_q=%d, absmax2=%d, code2=256, offset=1\n",
                (long)num_packed, num_blocks, num_groups);
        fprintf(stderr, "  Got:      packed=%zu, absmax_q=%zu, absmax2=%zu, code2=%zu, offset=%zu\n",
                r1, r2, r3, r4, r5);
        return EXIT_FAILURE;
    }

    printf("Offset: %.6f\n", h_offset);     

    // ── 2. Allocate device memory & copy ─────────────────────────────────────
    uint8_t* d_packed     = nullptr;
    uint8_t* d_absmax_q   = nullptr;
    half*    d_absmax2    = nullptr;
    half*    d_code2      = nullptr;
    half*    d_my_output  = nullptr;
    half*    d_bnb_output = nullptr;
    float*   d_absmax_decoded = nullptr;  // BNB 需要的单层量化 absmax (num_blocks * float)

    CHECK_CUDA(cudaMalloc(&d_packed,   num_packed));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, num_blocks));
    CHECK_CUDA(cudaMalloc(&d_absmax2,  num_groups * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_code2,    256 * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_my_output,   total * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_bnb_output,  total * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_absmax_decoded, num_blocks * sizeof(float)));

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
    int64_t grid_size_64 = (num_pairs + threads_per_block - 1) / threads_per_block;
    
    if (grid_size_64 > INT_MAX) {
        fprintf(stderr, "Error: Grid size overflow (%lld > INT_MAX). Data too large.\n",
                (long long)grid_size_64);
        return EXIT_FAILURE;
    }
    int grid_size = static_cast<int>(grid_size_64);

    // 创建流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // ── 4. Warmup ────────────────────────────────────────────────────────────
    printf("\n=== Warming up (%d iterations) ===\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        // 你的 kernel
        nf4_dequantize_kernel<<<grid_size, threads_per_block, 0, stream>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
            d_my_output, total, blocksize, group_size);
        // BNB kernel
        run_bnb_dequantize(
            d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
            d_absmax_decoded, d_bnb_output,
            total, num_blocks, blocksize, group_size, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    // ── 5. Timed iterations & Performance Comparison ────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    // 5.1 计时：你的 kernel
    CHECK_CUDA(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        nf4_dequantize_kernel<<<grid_size, threads_per_block, 0, stream>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
            d_my_output, total, blocksize, group_size);
    }
    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float my_elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&my_elapsed_ms, ev_start, ev_stop));
    float my_avg_ms = my_elapsed_ms / iters;

    // 5.2 计时：BNB kernel (只测量纯反量化时间，不包含双重量化absmax解码)
    // 预先解码 absmax（在计时循环外完成）
    {
        int threads = 256;
        int blocks_for_absmax = (num_blocks + threads - 1) / threads;
        decode_double_quant_absmax_kernel<<<blocks_for_absmax, threads, 0, stream>>>(
            d_absmax_q, d_absmax2, d_code2, h_offset,
            d_absmax_decoded, num_blocks, group_size
        );
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // 只计时 BNB 的纯反量化 kernel
    CHECK_CUDA(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        cdequantize_blockwise_fp16_nf4(
            nullptr,  // code (使用内部默认 NF4 表)
            const_cast<uint8_t*>(d_packed),
            d_absmax_decoded,
            d_bnb_output,
            blocksize,
            static_cast<int>(total),
            stream
        );
    }
    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float bnb_elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&bnb_elapsed_ms, ev_start, ev_stop));
    float bnb_avg_ms = bnb_elapsed_ms / iters;

    // 5.3 计算带宽和加速比
    double bytes_in  = static_cast<double>(num_packed + num_blocks
                                           + num_groups * 2 + 256 * 2);
    double bytes_out = static_cast<double>(total) * 2;
    double my_bw_gbs = (bytes_in + bytes_out) / (my_avg_ms * 1e6);
    double bnb_bw_gbs = (bytes_in + bytes_out) / (bnb_avg_ms * 1e6);
    float speedup = bnb_avg_ms / my_avg_ms;  // 加速比 = BNB耗时 / 你的耗时

    // 打印性能结果
    printf("\n=== Performance Comparison (My Kernel vs BNB) ===\n");
    printf("My Kernel  : %.4f ms (avg of %d iters) | Bandwidth: %.2f GB/s\n", 
           my_avg_ms, iters, my_bw_gbs);
    printf("BNB Kernel : %.4f ms (avg of %d iters) | Bandwidth: %.2f GB/s\n", 
           bnb_avg_ms, iters, bnb_bw_gbs);
    printf("Speedup    : %.2fx (My Kernel is faster than BNB)\n", speedup);

    // ── 6. Accuracy Check (集成到 CUDA 代码中) ──────────────────────────────
    compare_accuracy(d_my_output, d_bnb_output, total);

    // ── 7. Copy back & write output ──────────────────────────────────────────
    std::vector<uint16_t> h_output(total);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_my_output, total * sizeof(half),
                           cudaMemcpyDeviceToHost));

    int nonzero = 0;
    for (int64_t i = 0; i < 100 && i < total; ++i)
        if (h_output[i] != 0) nonzero++;
    printf("\nDEBUG: first 100 outputs, %d non-zero. raw[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
           nonzero, h_output[0], h_output[1], h_output[2], h_output[3]);

    FILE* fout = fopen(output_path, "wb");
    if (!fout) { perror("fopen(output)"); return EXIT_FAILURE; }
    size_t written = fwrite(h_output.data(), sizeof(uint16_t), total, fout);
    fclose(fout);

    if (written != static_cast<size_t>(total)) {
        fprintf(stderr, "Error: Failed to write all output data (wrote %zu of %ld)\n",
                written, (long)total);
        return EXIT_FAILURE;
    }

    printf("\nOutput: %ld fp16 values -> %s\n", (long)total, output_path);

    // ── 8. Cleanup ───────────────────────────────────────────────────────────
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_absmax_q));
    CHECK_CUDA(cudaFree(d_absmax2));
    CHECK_CUDA(cudaFree(d_code2));
    CHECK_CUDA(cudaFree(d_my_output));
    CHECK_CUDA(cudaFree(d_bnb_output));
    CHECK_CUDA(cudaFree(d_absmax_decoded));

    return 0;
}