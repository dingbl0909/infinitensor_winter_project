#!/usr/bin/env python3
"""
Test NF4 dequantize correctness against bitsandbytes reference.

Requirements (README.md 功能正确性 第三点):
  - 输出与 bitsandbytes 库的 dequantize_blockwise 结果对比
  - 平均绝对误差 (MAE) 小于 1e-2 (相对范围)
"""

import os
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F


# ----------------------------------------------------------------
# Helpers: binary I/O matching README spec
# ----------------------------------------------------------------

def write_input_bin(
    path: str,
    num_rows: int,
    num_cols: int,
    blocksize: int,
    packed_weights: np.ndarray,  # uint8
    absmax_q: np.ndarray,        # uint8
    absmax2: np.ndarray,         # float16
    code2: np.ndarray,           # float16[256]
    offset: float,
):
    """
    Write input binary file per README:
      [header]
        num_rows: int64
        num_cols: int64
        blocksize: int32
      [data]
        packed_weights: uint8[...]
        absmax_q: uint8[num_blocks]
        absmax2: float16[num_groups]
        code2: float16[256]
        offset: float32
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<q", num_rows))   # int64
        f.write(struct.pack("<q", num_cols))   # int64
        f.write(struct.pack("<i", blocksize))  # int32
        f.write(packed_weights.astype(np.uint8).tobytes())
        f.write(absmax_q.astype(np.uint8).tobytes())
        f.write(absmax2.astype(np.float16).tobytes())
        f.write(code2.astype(np.float16).tobytes())
        f.write(struct.pack("<f", offset))     # float32


def read_output_bin(path: str, num_elems: int, dtype=np.float16) -> np.ndarray:
    """Read dequantized output (fp16 or bf16 stored as uint16)."""
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=dtype)
    assert arr.shape[0] == num_elems, f"Expected {num_elems} elements, got {arr.shape[0]}"
    return arr


# ----------------------------------------------------------------
# bitsandbytes quantize -> our format conversion
# ----------------------------------------------------------------

def quantize_with_bnb(weight: torch.Tensor, blocksize: int = 64):
    """
    Quantize weight using bitsandbytes NF4 double quantization.
    Returns all components needed for our binary format.
    """
    weight_flat = weight.flatten().cuda().float()
    
    # quantize_4bit returns (packed_weights, quant_state)
    packed, state = F.quantize_4bit(
        weight_flat,
        blocksize=blocksize,
        quant_type="nf4",
        compress_statistics=True,  # enable double quantization
    )
    
    return packed, state


def extract_bnb_components(packed: torch.Tensor, state):
    """
    Extract components from bitsandbytes quant state for our binary format.
    """
    # packed_weights: uint8 tensor containing packed 4-bit indices
    packed_weights = packed.cpu().numpy()
    
    # For double quantization (compress_statistics=True):
    # state.absmax is quantized (uint8), state.state2 contains second-level info
    absmax_q = state.absmax.cpu().numpy().astype(np.uint8)
    
    # Second-level quantization state
    state2 = state.state2
    absmax2 = state2.absmax.cpu().numpy().astype(np.float16)
    code2 = state2.code.cpu().numpy().astype(np.float16)
    offset = float(state.offset.item()) if state.offset is not None else 0.0
    
    blocksize = state.blocksize
    group_size = state2.blocksize  # nested blocksize
    
    return {
        "packed_weights": packed_weights,
        "absmax_q": absmax_q,
        "absmax2": absmax2,
        "code2": code2,
        "offset": offset,
        "blocksize": blocksize,
        "group_size": group_size,
    }


def dequantize_with_bnb(packed: torch.Tensor, state) -> torch.Tensor:
    """Dequantize using bitsandbytes as reference."""
    return F.dequantize_4bit(packed, state)


# ----------------------------------------------------------------
# Test runner
# ----------------------------------------------------------------

def find_executable() -> Path:
    """Find the compiled nf4_dequantize executable."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Try build/nf4_dequantize (Makefile default)
    exe = project_dir / "build" / "nf4_dequantize"
    if exe.exists():
        return exe
    
    raise FileNotFoundError(
        f"Cannot find nf4_dequantize executable. "
        f"Please run 'make' in {project_dir}"
    )


def run_dequantize(
    exe_path: Path,
    input_bin: str,
    output_bin: str,
    group_size: int = 256,
) -> None:
    """Run the CUDA dequantize executable."""
    cmd = [
        str(exe_path),
        "--input", input_bin,
        "--output", output_bin,
        "--group_size", str(group_size),
        "--warmup", "1",
        "--iters", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nf4_dequantize failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


def parse_cuda_benchmark(stdout: str):
    """
    Parse kernel timing from CUDA executable output.
    Expected line:
      My Kernel  : X.XXXX ms ...
    """
    my_match = re.search(r"My Kernel\s*:\s*([0-9.]+)\s*ms", stdout)
    if not my_match:
        raise ValueError("Failed to parse 'My Kernel' timing from CUDA output")
    return {"my_kernel_ms": float(my_match.group(1))}


def benchmark_bnb_python(
    packed: torch.Tensor,
    state,
    warmup: int = 20,
    iters: int = 200,
):
    """Benchmark bitsandbytes dequantize_4bit in Python (CUDA events)."""
    # Warmup
    for _ in range(warmup):
        _ = F.dequantize_4bit(packed, state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = F.dequantize_4bit(packed, state)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / iters


def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))


def compute_relative_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Compute MAE relative to the value range."""
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    mae = np.mean(np.abs(a_f - b_f))
    value_range = max(np.abs(b_f).max(), 1e-8)
    return mae / value_range


# ----------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------

def test_correctness(
    num_rows: int,
    num_cols: int,
    blocksize: int = 64,
    benchmark: bool = True,
    benchmark_warmup: int = 20,
    benchmark_iters: int = 200,
    seed: int = 42,
):
    """
    Test that our CUDA implementation matches bitsandbytes within MAE < 1e-2.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {num_rows}x{num_cols}, blocksize={blocksize}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random weights (simulating typical model weights)
    weight = torch.randn(num_rows, num_cols, dtype=torch.float32)
    
    # Quantize with bitsandbytes
    print("Quantizing with bitsandbytes...")
    packed, state = quantize_with_bnb(weight, blocksize=blocksize)
    
    # Get reference dequantized output
    print("Dequantizing with bitsandbytes (reference)...")
    ref_output = dequantize_with_bnb(packed, state)
    ref_output_np = ref_output.cpu().numpy().astype(np.float16)
    
    # Extract components for our format
    components = extract_bnb_components(packed, state)
    
    print(f"  packed_weights shape: {components['packed_weights'].shape}")
    print(f"  absmax_q shape: {components['absmax_q'].shape}")
    print(f"  absmax2 shape: {components['absmax2'].shape}")
    print(f"  code2 shape: {components['code2'].shape}")
    print(f"  offset: {components['offset']}")
    print(f"  blocksize: {components['blocksize']}")
    print(f"  group_size: {components['group_size']}")
    
    # Find executable
    exe_path = find_executable()
    print(f"Using executable: {exe_path}")
    
    # Run our CUDA implementation
    with tempfile.TemporaryDirectory() as tmpdir:
        input_bin = os.path.join(tmpdir, "input.bin")
        output_bin = os.path.join(tmpdir, "output.bin")
        
        # Write input binary
        write_input_bin(
            input_bin,
            num_rows=num_rows,
            num_cols=num_cols,
            blocksize=components["blocksize"],
            packed_weights=components["packed_weights"],
            absmax_q=components["absmax_q"],
            absmax2=components["absmax2"],
            code2=components["code2"],
            offset=components["offset"],
        )
        
        # Run CUDA dequantize
        print("Running CUDA nf4_dequantize...")
        cuda_stdout = run_dequantize(
            exe_path,
            input_bin,
            output_bin,
            group_size=components["group_size"],
        )
        print(cuda_stdout)
        
        # Read output
        num_elems = num_rows * num_cols
        our_output_np = read_output_bin(output_bin, num_elems, dtype=np.float16)
    
    # Compute errors
    mae = compute_mae(our_output_np, ref_output_np)
    relative_mae = compute_relative_mae(our_output_np, ref_output_np)
    max_diff = np.max(np.abs(our_output_np.astype(np.float32) - ref_output_np.astype(np.float32)))
    
    print(f"\nResults:")
    print(f"  MAE (absolute):  {mae:.6f}")
    print(f"  MAE (relative):  {relative_mae:.6f}")
    print(f"  Max diff:        {max_diff:.6f}")
    print(f"  Reference range: [{ref_output_np.min():.4f}, {ref_output_np.max():.4f}]")
    print(f"  Our range:       [{our_output_np.min():.4f}, {our_output_np.max():.4f}]")
    
    # Check requirement: MAE < 1e-2 (relative range)
    threshold = 1e-2
    passed = relative_mae < threshold
    
    if passed:
        print(f"\n✓ PASSED: relative MAE ({relative_mae:.6f}) < {threshold}")
    else:
        print(f"\n✗ FAILED: relative MAE ({relative_mae:.6f}) >= {threshold}")

    benchmark_metrics = {}
    if benchmark:
        print(f"\n{'-'*60}")
        print("Performance benchmark (Python orchestration)")
        print(f"{'-'*60}")
        # 1) Parse our kernel timing from CUDA executable output
        cuda_bench = parse_cuda_benchmark(cuda_stdout)
        my_ms = cuda_bench["my_kernel_ms"]

        # 2) Time bitsandbytes dequantize in Python with same quantized data
        bnb_py_ms = benchmark_bnb_python(
            packed=packed,
            state=state,
            warmup=benchmark_warmup,
            iters=benchmark_iters,
        )
        py_speedup = bnb_py_ms / my_ms

        benchmark_metrics = {
            "my_kernel_ms": my_ms,
            "bnb_python_ms": bnb_py_ms,
            "speedup_vs_bnb_python": py_speedup,
        }

        print(f"My kernel (CUDA exe):     {my_ms:.4f} ms")
        print(f"BNB kernel (Python API):  {bnb_py_ms:.4f} ms")
        print(f"Speedup (BNB_py / My):    {py_speedup:.2f}x")

    return passed, {
        "mae": mae,
        "relative_mae": relative_mae,
        "max_diff": max_diff,
        **benchmark_metrics,
    }


def main():
    """Run all test cases."""
    print("=" * 60)
    print("NF4 Dequantize Correctness Test")
    print("Requirement: MAE < 1e-2 (relative to value range)")
    print("=" * 60)
    
    test_cases = [
        # (num_rows, num_cols, blocksize)
        # Standard aligned cases
        (64, 64, 64),
        (128, 256, 64),
        (512, 1024, 64),
        
        # # Non-aligned cases (boundary handling)
        (100, 100, 64),      # Not aligned to blocksize
        (127, 255, 64),      # Odd dimensions
        (1, 1000, 64),       # Single row
        (1000, 1, 64),       # Single column
        (33, 77, 64),        # Arbitrary odd dimensions
        
        # # Different blocksizes
        (256, 256, 128),
    ]
    
    results = []
    for num_rows, num_cols, blocksize in test_cases:
        try:
            passed, metrics = test_correctness(
                num_rows=num_rows,
                num_cols=num_cols,
                blocksize=blocksize,
                benchmark=True,
                benchmark_warmup=20,
                benchmark_iters=200,
            )
            results.append((num_rows, num_cols, blocksize, passed, metrics))
        except Exception as e:
            print(f"\n✗ ERROR for {num_rows}x{num_cols}: {e}")
            results.append((num_rows, num_cols, blocksize, False, {"error": str(e)}))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for r in results if r[3])
    
    for num_rows, num_cols, blocksize, ok, metrics in results:
        status = "PASS" if ok else "FAIL"
        if "error" in metrics:
            print(f"  [{status}] {num_rows:4d}x{num_cols:4d} bs={blocksize:3d}: ERROR - {metrics['error']}")
        else:
            perf = ""
            if "speedup_vs_bnb_python" in metrics:
                perf = (
                    f", My={metrics['my_kernel_ms']:.4f}ms, "
                    f"BNB_py={metrics['bnb_python_ms']:.4f}ms, "
                    f"Speedup={metrics['speedup_vs_bnb_python']:.2f}x"
                )
            print(
                f"  [{status}] {num_rows:4d}x{num_cols:4d} bs={blocksize:3d}: "
                f"MAE(rel)={metrics['relative_mae']:.6f}{perf}"
            )
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())