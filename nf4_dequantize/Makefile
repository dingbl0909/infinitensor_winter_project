NVCC ?= /usr/local/cuda/bin/nvcc

# Set SM to your GPU arch (T4: 75). Override like: make SM=80
SM ?= 86

TARGET := nf4_dequantize
SRC := src/nf4_dequantize.cu
BUILD_DIR := build
BIN := $(BUILD_DIR)/$(TARGET)

# 基础编译参数（保留你的原有配置）
NVCCFLAGS ?= -O3 --use_fast_math -std=c++17
NVCCFLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)
# 补充：让 NVCC 传递 C++ 链接参数到 ld
NVCCFLAGS += -Xcompiler -fPIC

# ===================== 精准适配 bitsandbytes 库路径 =====================
PYTHON := /usr/bin/python3

# bitsandbytes 安装路径
BNB_PATH := $(shell $(PYTHON) -c "import bitsandbytes; print(bitsandbytes.__path__[0])" 2>/dev/null)
ifeq ($(BNB_PATH),)
BNB_PATH := /usr/local/lib/python3.12/dist-packages/bitsandbytes
endif

# 指定 CUDA 版本的 bitsandbytes 库（根据你的系统 CUDA 版本选）
# 先执行：nvcc --version 查看系统 CUDA 版本，再修改
BNB_CUDA_VERSION := 128
BNB_LIB_NAME := libbitsandbytes_cuda$(BNB_CUDA_VERSION).so
BNB_LIB_PATH := $(BNB_PATH)/$(BNB_LIB_NAME)

# 检查指定的 CUDA 版本库是否存在
ifeq (,$(wildcard $(BNB_LIB_PATH)))
$(error "bitsandbytes CUDA $(BNB_CUDA_VERSION) 库不存在！请修改 BNB_CUDA_VERSION 为以下值之一：118/120/121/122/123/124/125/126/128/129/130")
endif

# 添加 bitsandbytes 头文件路径
NVCCFLAGS += -I$(BNB_PATH)/include

# 链接参数：直接指向 bitsandbytes 根目录（库文件在这里）
# -l 参数：去掉 lib 前缀和 .so 后缀，比如 libbitsandbytes_cuda122.so → bitsandbytes_cuda122
LDFLAGS += -L$(BNB_PATH) -lbitsandbytes_cuda$(BNB_CUDA_VERSION)
# 运行时库路径（嵌入到可执行文件）
LDFLAGS += -Xlinker -rpath=$(BNB_PATH)
# 补充依赖库
LDFLAGS += -lcudart -lpthread -ldl

PROFILE_BIN := $(BUILD_DIR)/$(TARGET)_profile

.PHONY: all clean run help profile

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译命令（最终版）：将 LDFLAGS 放在 $< 之后（NVCC 要求链接参数在源文件后）
$(BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

# Profile build: 带 -lineinfo 用于 ncu 源代码级分析
$(PROFILE_BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -lineinfo -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

run: $(BIN)
	@echo "Running NF4 dequantize benchmark (CUDA $(BNB_CUDA_VERSION))..."
	./$(BIN) --input data/input.bin --output data/output.bin --group_size 256 --warmup 5 --iters 20

profile: $(PROFILE_BIN)
	@echo "=== Running ncu profile on nf4_dequantize_kernel ==="
	ncu --set full \
	    --kernel-name nf4_dequantize_kernel \
	    --launch-skip 0 --launch-count 1 \
	    -o $(BUILD_DIR)/nf4_profile \
	    ./$(PROFILE_BIN) --input data/input.bin --output data/output.bin --group_size 256 --warmup 0 --iters 1
	@echo "Profile saved to $(BUILD_DIR)/nf4_profile.ncu-rep"
	@echo "Open with: ncu-ui $(BUILD_DIR)/nf4_profile.ncu-rep"

help:
	@echo "Targets:"
	@echo "  make [SM=75]            Build $(BIN) (link with bitsandbytes CUDA $(BNB_CUDA_VERSION))"
	@echo "  make clean              Remove build/"
	@echo "  make run                Build and run with test parameters"
	@echo ""
	@echo "  make profile             Build with -lineinfo and run ncu profiling"
	@echo ""
	@echo "修改 CUDA 版本：编辑 Makefile 中的 BNB_CUDA_VERSION 变量（当前：$(BNB_CUDA_VERSION)）"
	@echo "Examples:"
	@echo "  make SM=75              # T4"
	@echo "  make SM=80              # A100"
	@echo "  make SM=86              # RTX 30xx"
	@echo "  make SM=89              # RTX 40xx"
	@echo "  make SM=90              # H100"