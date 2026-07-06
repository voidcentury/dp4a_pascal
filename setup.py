"""
setup.py — builds dp4a_ext CUDA extension for sm_61 (Pascal)

Usage:
    python setup.py build_ext --inplace

After build, import with: import dp4a_ext
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")

cxx_flags = ["-O3", "-std=c++17"]
nvcc_flags = [
    "-O3",
    "-arch=sm_61",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "-Wno-deprecated-gpu-targets",
    "--ptxas-options=-v",
]

# Suppress specific nvcc warnings
if sys.platform == "win32":
    nvcc_flags += ["-Xcudafe", "--diag_suppress=186"]
    cxx_flags = [f"/O2", "/std:c++17"]

setup(
    name="dp4a_ext",
    version="0.1.0",
    description="DP4A INT8 GEMM CUDA extension for Pascal GPUs",
    ext_modules=[
        CUDAExtension(
            name="dp4a_ext",
            sources=[
                "dp4a_bindings.cpp",
                "dp4a_block.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
