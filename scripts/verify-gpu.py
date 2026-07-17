#!/usr/bin/env python3
"""Verify the PyTorch installation and, when available, CUDA execution."""

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="succeed after CPU import checks when no CUDA device is available",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device is visible to PyTorch.", file=sys.stderr)
        return 0 if args.allow_no_cuda else 1

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {props.name}; compute capability "
            f"{props.major}.{props.minor}; VRAM {props.total_memory / 2**30:.1f} GiB"
        )

    device = torch.device("cuda:0")
    a = torch.randn((512, 512), device=device)
    b = torch.randn((512, 512), device=device)
    result = a @ b
    torch.cuda.synchronize(device)
    if result.shape != (512, 512) or not torch.isfinite(result).all().item():
        print("CUDA matrix multiplication returned an invalid result.", file=sys.stderr)
        return 1
    print(f"CUDA matrix multiplication: PASS ({result.shape[0]}x{result.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
