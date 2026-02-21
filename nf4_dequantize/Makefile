NVCC ?= /usr/local/cuda/bin/nvcc

# Set SM to your GPU arch (T4: 75). Override like: make SM=80
SM ?= 89

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
# 自定义 Python 环境路径
PY_ENV_PATH := /home/blding/pytorch_env
PYTHON := $(PY_ENV_PATH)/bin/python3

# 自动查找 bitsandbytes 安装路径
BNB_PATH := $(shell $(PYTHON) -c "import bitsandbytes; print(bitsandbytes.__path__[0])" 2>/dev/null)

# 检查 bitsandbytes 是否安装
ifeq ($(BNB_PATH),)
$(error "bitsandbytes not found in $(PY_ENV_PATH)! Please install it first: $(PYTHON) -m pip install bitsandbytes")
endif

# 指定 CUDA 版本的 bitsandbytes 库（根据你的系统 CUDA 版本选）
# 先执行：nvcc --version 查看系统 CUDA 版本，再修改
BNB_CUDA_VERSION := 126
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

.PHONY: all clean run help

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译命令（最终版）：将 LDFLAGS 放在 $< 之后（NVCC 要求链接参数在源文件后）
$(BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

run: $(BIN)
	@echo "Running NF4 dequantize benchmark (CUDA $(BNB_CUDA_VERSION))..."
	./$(BIN) --input test_quant.bin --output test_dequant.bin --group_size 256 --warmup 5 --iters 20

help:
	@echo "Targets:"
	@echo "  make [SM=75]            Build $(BIN) (link with bitsandbytes CUDA $(BNB_CUDA_VERSION))"
	@echo "  make clean              Remove build/"
	@echo "  make run                Build and run with test parameters"
	@echo ""
	@echo "修改 CUDA 版本：编辑 Makefile 中的 BNB_CUDA_VERSION 变量（当前：$(BNB_CUDA_VERSION)）"
	@echo "Examples:"
	@echo "  make SM=75              # T4"
	@echo "  make SM=80              # A100"
	@echo "  make SM=86              # RTX 30xx"
	@echo "  make SM=89              # RTX 40xx"
	@echo "  make SM=90              # H100"