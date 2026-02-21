#!/usr/bin/env python3
"""
Generate NF4 input binary for nf4_dequantize.
Uses bitsandbytes to quantize random weights and writes the binary per README spec.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure we can import from same directory
sys.path.insert(0, str(Path(__file__).parent))
from test_correctness import (
    extract_bnb_components,
    quantize_with_bnb,
    write_input_bin,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate NF4 input binary for nf4_dequantize"
    )
    parser.add_argument(
        "-o", "--output",
        default="input.bin",
        help="Output binary path (default: input.bin)",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=256,
        help="Number of rows (default: 128)",
    )
    parser.add_argument(
        "--num_cols",
        type=int,
        default=256,
        help="Number of columns (default: 256)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=64,
        help="Quantization blocksize (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Generating: {args.num_rows}x{args.num_cols}, blocksize={args.blocksize}")
    weight = torch.randn(args.num_rows, args.num_cols, dtype=torch.float32)

    print("Quantizing with bitsandbytes (NF4 double quant)...")
    packed, state = quantize_with_bnb(weight, blocksize=args.blocksize)
    components = extract_bnb_components(packed, state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {output_path}")
    write_input_bin(
        str(output_path),
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        blocksize=components["blocksize"],
        packed_weights=components["packed_weights"],
        absmax_q=components["absmax_q"],
        absmax2=components["absmax2"],
        code2=components["code2"],
        offset=components["offset"],
    )
    print(f"Done. group_size={components['group_size']} (for --group_size when running dequant)")


if __name__ == "__main__":
    main()
