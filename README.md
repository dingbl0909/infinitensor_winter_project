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

## 编译与运行指南

### 环境要求

- **CUDA Toolkit**：支持的版本 11.8 - 13.0
- **GPU**：NVIDIA GPU with compute capability ≥ 7.5（T4 及以上）
- **Python 环境**：Python 3.8+ （用于测试和对比）
- **依赖库**：bitsandbytes（用于正确性验证和性能对比）

### 安装依赖

```bash
# 安装 bitsandbytes 库
pip install bitsandbytes

# 或在指定 Python 环境中安装
/path/to/your/python -m pip install bitsandbytes
```

### 编译步骤

项目使用 Makefile 进行构建，编译前需要配置以下参数：

#### 1. 配置 GPU 架构（SM 版本）

根据你的 GPU 型号设置 `SM` 参数：

| GPU 型号 | SM 版本 | 编译命令 |
|---------|--------|---------|
| T4      | 75     | `make SM=75` |
| A100    | 80     | `make SM=80` |
| RTX 30xx| 86     | `make SM=86` |
| RTX 40xx| 89     | `make SM=89` |
| H100    | 90     | `make SM=90` |

#### 2. 配置 Python 环境路径（可选）

如果你的 Python 环境不在默认位置，需要编辑 `nf4_dequantize/Makefile` 第 19 行：

```makefile
PY_ENV_PATH := /path/to/your/python_env
```

#### 3. 配置 CUDA 版本（可选）

Makefile 会自动检测 bitsandbytes 支持的 CUDA 版本。如需手动指定，编辑 Makefile 第 32 行：

```makefile
BNB_CUDA_VERSION := 126  # 可选：118/120/121/122/123/124/125/126/128/129/130
```

#### 4. 执行编译

```bash
# 进入项目目录
cd nf4_dequantize

# 查看帮助信息
make help

# 编译（使用默认 SM=89）
make

# 或指定 GPU 架构编译（例如 T4）
make SM=75

# 清理构建产物
make clean
```

编译成功后会生成可执行文件：`build/nf4_dequantize`

### 运行程序

#### 基本用法

```bash
./build/nf4_dequantize --input <INPUT_BIN> --output <OUTPUT_BIN> [OPTIONS]
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--input` | 输入的量化权重二进制文件路径（必需） | - |
| `--output` | 输出的反量化结果文件路径（必需） | - |
| `--group_size` | 二级量化的组大小 | 256 |
| `--warmup` | 预热迭代次数 | 3 |
| `--iters` | 性能测试迭代次数 | 10 |

#### 运行示例

```bash
# 使用示例数据运行
./build/nf4_dequantize --input ./data/input.bin --output ./data/output.bin --group_size 256 --warmup 5 --iters 20

# 快速测试（使用 make run）
make run
```

#### 输出示例

```
Shape: 4096x11008  blocksize=64  group_size=256
Elements=45088768  packed_bytes=22544384  blocks=704512  groups=2752
Offset: 0.000000
Kernel time : 0.3456 ms (avg of 20 iters)
Bandwidth   : 267.89 GB/s
```

### 性能测试

项目提供了与 bitsandbytes 库的性能对比脚本：

```bash
# 进入测试目录
cd tests

# 运行正确性测试
python test_correctness.py

# 运行性能对比测试（需要先生成测试数据）
python gen_input_bin.py  # 生成测试输入
python speedup_rate_test.py  # 对比性能
```

### 使用 Nsight Systems 性能分析

```bash
# 分析你的实现
sudo nsys profile --trace=cuda --sample=none -o my_profile \
  ./build/nf4_dequantize --input ./data/input.bin --output ./data/output.bin --warmup 5 --iters 20

# 查看 GPU kernel 统计
nsys stats --report cuda_gpu_kern_sum my_profile.nsys-rep

# 使用 GUI 查看详细报告（在图形界面环境）
nsys-ui my_profile.nsys-rep
```

### 故障排查

#### 编译错误：找不到 bitsandbytes

```bash
# 检查 bitsandbytes 是否正确安装
python -c "import bitsandbytes; print(bitsandbytes.__path__)"

# 如果未安装，执行：
pip install bitsandbytes
```

#### 编译错误：CUDA 版本不匹配

错误信息：`bitsandbytes CUDA XXX 库不存在`

解决方法：
1. 查看你的 CUDA 版本：`nvcc --version`
2. 编辑 Makefile，修改 `BNB_CUDA_VERSION` 为对应值
3. 重新编译：`make clean && make`

#### 运行时错误：找不到共享库

```bash
# 检查 bitsandbytes 库路径
ldd build/nf4_dequantize | grep bitsandbytes

# 如果显示 "not found"，手动设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/bitsandbytes:$LD_LIBRARY_PATH
```
