#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>

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

// NF4 码本
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

// NF4 解量化核
// 优化点：
//   1) shared memory 预加载 scale，消除 global load broadcast 浪费
//   2) NF4 表从 constant → shared memory，避免 constant cache 序列化
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
    extern __shared__ float smem[];
    float* s_nf4    = smem;            // [16]
    float* s_scales = smem + 16;       // [num_blks_in_tb]

    if (threadIdx.x < 16) s_nf4[threadIdx.x] = c_nf4[threadIdx.x];

    // 协作预加载当前 threadblock 覆盖的 scale
    int base_elem = static_cast<int>(static_cast<int64_t>(blockIdx.x) * blockDim.x * 2);
    int base_blk  = base_elem / blocksize;
    int end_elem  = base_elem + static_cast<int>(blockDim.x) * 2;
    if (end_elem > static_cast<int>(total_elements))
        end_elem = static_cast<int>(total_elements);
    int end_blk  = (end_elem - 1) / blocksize;
    int num_blks = end_blk - base_blk + 1;

    if (static_cast<int>(threadIdx.x) < num_blks) {
        int blk = base_blk + threadIdx.x;
        int grp = blk / group_size;
        s_scales[threadIdx.x] = __half2float(code2[absmax_q[blk]])
                               * __half2float(absmax2[grp])
                               + offset;
    }
    __syncthreads();

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t num_pairs = (total_elements + 1) / 2;
    if (tid >= num_pairs) return;

    int64_t elem0 = tid * 2;

    uint8_t packed   = packed_weights[tid];
    uint8_t idx_even = (packed >> 4) & 0x0F;
    uint8_t idx_odd  = packed & 0x0F;

    int blk0  = static_cast<int>(elem0 / blocksize);
    float sc0 = s_scales[blk0 - base_blk];

    half h0 = __float2half(s_nf4[idx_even] * sc0);

    if (elem0 + 1 < total_elements) {
        int blk1 = static_cast<int>((elem0 + 1) / blocksize);
        float sc1 = (blk1 != blk0) ? s_scales[blk1 - base_blk] : sc0;
        half h1 = __float2half(s_nf4[idx_odd] * sc1);

        uint32_t packed_out = static_cast<uint32_t>(__half_as_ushort(h0))
                            | (static_cast<uint32_t>(__half_as_ushort(h1)) << 16);
        reinterpret_cast<uint32_t*>(output)[tid] = packed_out;
    } else {
        output[elem0] = h0;
    }
}

// CLI helpers
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

int main(int argc, char** argv) {
    const char* input_path  = get_arg(argc, argv, "--input");
    const char* output_path = get_arg(argc, argv, "--output");
    int group_size = get_int_arg(argc, argv, "--group_size", 256);
    int warmup     = get_int_arg(argc, argv, "--warmup", 3);
    int iters      = get_int_arg(argc, argv, "--iters", 10);

    if (!input_path || !output_path) {
        fprintf(stderr,
                "Usage: %s --input BIN --output BIN "
                "[--group_size N] [--warmup N] [--iters N]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    // 1. 读取二进制输入文件
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

    int64_t total      = num_rows * num_cols;
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

    size_t r1 = fread(h_packed.data(),   1,                num_packed,  fin);
    size_t r2 = fread(h_absmax_q.data(), 1,                num_blocks,  fin);
    size_t r3 = fread(h_absmax2.data(),  sizeof(uint16_t), num_groups,  fin);
    size_t r4 = fread(h_code2.data(),    sizeof(uint16_t), 256,         fin);
    size_t r5 = fread(&h_offset,         sizeof(float),    1,           fin);
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

    // 2. 分配设备内存并拷贝数据
    uint8_t* d_packed   = nullptr;
    uint8_t* d_absmax_q = nullptr;
    half*    d_absmax2  = nullptr;
    half*    d_code2    = nullptr;
    half*    d_output   = nullptr;

    CHECK_CUDA(cudaMalloc(&d_packed,   num_packed));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, num_blocks));
    CHECK_CUDA(cudaMalloc(&d_absmax2,  num_groups * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_code2,    256 * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output,   total * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_packed,   h_packed.data(),   num_packed,              cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), num_blocks,              cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2,  h_absmax2.data(),  num_groups * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2,    h_code2.data(),    256 * sizeof(half),      cudaMemcpyHostToDevice));

    // 3. 配置 launch 参数
    const int threads_per_block = 256;
    int64_t   num_pairs    = (total + 1) / 2;
    int64_t   grid_size_64 = (num_pairs + threads_per_block - 1) / threads_per_block;

    if (grid_size_64 > INT_MAX) {
        fprintf(stderr, "Error: Grid size overflow (%lld > INT_MAX). Data too large.\n",
                (long long)grid_size_64);
        return EXIT_FAILURE;
    }
    int grid_size = static_cast<int>(grid_size_64);

    // shared memory: 16 floats (NF4 表) + 每个 threadblock 最多覆盖的块数
    int    max_blks_per_tb = (threads_per_block * 2 + blocksize - 1) / blocksize + 1;
    size_t smem_bytes      = (16 + max_blks_per_tb) * sizeof(float);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 4. Warmup
    printf("\n=== Warming up (%d iterations) ===\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        nf4_dequantize_kernel<<<grid_size, threads_per_block, smem_bytes, stream>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
            d_output, total, blocksize, group_size);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    // 5. 计时
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        nf4_dequantize_kernel<<<grid_size, threads_per_block, smem_bytes, stream>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
            d_output, total, blocksize, group_size);
    }
    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    float avg_ms = elapsed_ms / iters;

    double bytes_in  = static_cast<double>(num_packed + num_blocks
                                           + num_groups * 2 + 256 * 2);
    double bytes_out = static_cast<double>(total) * 2;
    double bw_gbs    = (bytes_in + bytes_out) / (avg_ms * 1e6);

    printf("\n=== Performance ===\n");
    printf("My Kernel  : %.4f ms (avg of %d iters) | Bandwidth: %.2f GB/s\n",
           avg_ms, iters, bw_gbs);

    // 6. 拷回结果并写文件
    std::vector<uint16_t> h_output(total);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, total * sizeof(half),
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

    // 7. 释放资源
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_absmax_q));
    CHECK_CUDA(cudaFree(d_absmax2));
    CHECK_CUDA(cudaFree(d_code2));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
