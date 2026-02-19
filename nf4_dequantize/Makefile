NVCC ?= /usr/local/cuda/bin/nvcc

# Set SM to your GPU arch (T4: 75). Override like: make SM=80
SM ?= 89

TARGET := nf4_dequantize
SRC := src/nf4_dequantize.cu
BUILD_DIR := build
BIN := $(BUILD_DIR)/$(TARGET)

NVCCFLAGS ?= -O3 --use_fast_math -std=c++17
NVCCFLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)

.PHONY: all clean run help

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make [SM=75]            Build $(BIN)"
	@echo "  make clean              Remove build/"
	@echo ""
	@echo "Examples:"
	@echo "  make SM=75              # T4"
	@echo "  make SM=80              # A100"
	@echo "  make SM=86              # RTX 30xx"
	@echo "  make SM=89              # RTX 40xx"

