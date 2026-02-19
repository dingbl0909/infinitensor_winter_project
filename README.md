# infinitensor_winter_project

## 1. NF4 解量化算子开发

### 任务内容

开发一个 CUDA 程序，实现单核的 NF4 解量化算子，将压缩后的 4-bit 权重实时解压为 16-bit 浮点（BF16 或 FP16）输出。

### 输入文件定义

#### 1. 权重数据文件（二进制格式）

```
[header]
num_rows: int64               # 权重矩阵行数
num_cols: int64               # 权重矩阵列数
blocksize: int32              # 块大小（通常为 64 或 128）

[data]
packed_weights: uint8[num_rows * num_cols / 2]   # 每字节存两个 4-bit 索引
absmax_q: uint8[num_blocks]                      # 一级量化缩放因子（每块一个）
absmax2: float16[num_groups]                     # 二级缩放因子（每组一个）
code2: float16[256]                              # 二级码表
offset: float32                                  # 量化偏移（通常为 0）
```

#### 2. 参数文件（文本格式）

```
blocksize = 64                 # 块大小，必须与数据文件一致
compute_type = "bf16"          # 输出数据类型：bf16 或 fp16
target_gpu = "T4"              # 目标 GPU 型号（影响优化策略）
```

### 输出文件定义

- **解量化后的权重**：二进制文件，按行主序存储的 bfloat16 或 float16 数组，形状为 `[num_rows, num_cols]`。
- **性能日志**：核函数执行时间（ms）、有效内存带宽（GB/s）、相比 bitsandbytes 参考实现的加速比。

### 要求

1. **功能正确性**
   - 正确实现 NF4 索引到 FP16 的映射（NF4 查找表预定义为 16 个常数，需硬编码在 Kernel 中）
   - 正确实现两级缩放的解量化公式
   - 输出与 bitsandbytes 库的 `dequantize_blockwise` 结果对比，平均绝对误差（MAE）小于 1e-2（相对范围）

2. **4-bit 解包与内存访问**
   - 必须实现向量化内存访问（Packed Store）：每个线程一次处理两个 4-bit 索引，计算两个 BF16 值后打包成一个 32-bit `uint32_t` 一次性写入全局内存

3. **边界处理**
   - 支持任意形状的矩阵（行数和列数无需对齐到块大小的整数倍）
   - 处理最后一行的部分块时需正确控制边界

4. **平台适配**：默认需在英伟达平台上支持。每在一款国产平台上支持，额外加 20% 乘算系数。
