#!/usr/bin/env python3
"""Generic ATS 4.6 HALO orchestration driven by the ATS 4.8 CUDA core."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import ATS_v4_6_halo as halo


HERE = Path(__file__).resolve().parent
CORE = HERE / "ATS_v4_8_cuda.exe"
PACKAGED_CORES = {
    (8, 6): HERE / "ATS_v4_8_cuda_sm86.exe",
    (8, 9): HERE / "ATS_v4_8_cuda_sm89.exe",
}


def packaged_cuda_core() -> tuple[Path, str]:
    """Select a packaged kernel matching device 0, or force the CPU scorer."""
    try:
        driver = ctypes.WinDLL("nvcuda.dll")
        driver.cuInit.argtypes = (ctypes.c_uint,)
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceComputeCapability.argtypes = (
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        )
        driver.cuDeviceComputeCapability.restype = ctypes.c_int
        device = ctypes.c_int()
        major = ctypes.c_int()
        minor = ctypes.c_int()
        if driver.cuInit(0) != 0 or driver.cuDeviceGet(ctypes.byref(device), 0) != 0:
            raise OSError("CUDA driver did not expose device 0")
        if driver.cuDeviceComputeCapability(
            ctypes.byref(major), ctypes.byref(minor), device.value,
        ) != 0:
            raise OSError("CUDA driver did not report compute capability")
        capability = (major.value, minor.value)
        matching_core = PACKAGED_CORES.get(capability)
        if matching_core is not None and matching_core.exists():
            print(
                f"HALO_V48_CUDA_PREFLIGHT compute_capability={major.value}.{minor.value} "
                f"packaged_kernel=sm_{major.value}{minor.value} mode=auto "
                f"core={matching_core.name}",
                flush=True,
            )
            return matching_core, "auto"
        print(
            f"HALO_V48_CUDA_PREFLIGHT compute_capability={major.value}.{minor.value} "
            "mode=cpu reason=matching_kernel_not_built",
            flush=True,
        )
        return CORE, "off"
    except (AttributeError, OSError) as exc:
        print(
            "HALO_V48_CUDA_PREFLIGHT mode=cpu "
            f"reason=capability_probe_failed detail={exc}",
            flush=True,
        )
        return CORE, "off"


def main() -> int:
    args = halo.build_parser().parse_args()
    try:
        args.seed = args.seed.resolve()
        args.output_dir = args.output_dir.resolve()
        halo.CORE, halo.CORE_CUDA_MODE = packaged_cuda_core()
        return halo.run_search(args)
    except Exception as exc:
        print(f"HALO_V48_STAGE_ERROR {exc}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
