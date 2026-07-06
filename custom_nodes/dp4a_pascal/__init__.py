import os
import sys
import logging

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))

_DP4A_AVAILABLE = False
_IS_PASCAL = False

# ── GPU capability check ───────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        _IS_PASCAL = (cap[0] == 6 and cap[1] == 1)
        if _IS_PASCAL:
            logging.info("[DP4A_Pascal] Detected sm_%d%d GPU: %s",
                         cap[0], cap[1], torch.cuda.get_device_name(0))
        else:
            logging.info("[DP4A_Pascal] GPU sm_%d%d is not Pascal (sm_61); DP4A kernel will not be applied",
                         cap[0], cap[1])
    else:
        logging.info("[DP4A_Pascal] No CUDA GPU available")
except Exception:
    pass

# ── Load dp4a_ext ──────────────────────────────────────────────────────
if _IS_PASCAL:
    try:
        import torch
        import ctypes

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        torch_libs = [
            "libc10.so", "libc10_cuda.so", "libtorch.so",
            "libtorch_cpu.so", "libtorch_python.so", "libtorch_cuda.so",
        ]
        for lib_name in torch_libs:
            lib_path = os.path.join(torch_lib, lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

        sys.path.insert(0, _NODE_DIR)
        import dp4a_ext
        _DP4A_AVAILABLE = True
        logging.info("[DP4A_Pascal] dp4a_ext loaded successfully from %s", dp4a_ext.__file__)
    except Exception as e:
        logging.warning("[DP4A_Pascal] Failed to load dp4a_ext: %s", e)
        _DP4A_AVAILABLE = False

from .node import DP4APascalApply

NODE_CLASS_MAPPINGS = {
    "DP4APascalApply": DP4APascalApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DP4APascalApply": "Apply DP4A Pascal INT8 Kernel",
}

if _IS_PASCAL and _DP4A_AVAILABLE:
    logging.info("[DP4A_Pascal] Node registered — DP4A kernel ready on Pascal GPU")
elif _IS_PASCAL and not _DP4A_AVAILABLE:
    logging.warning("[DP4A_Pascal] Node registered on Pascal GPU but dp4a_ext could not be loaded")
else:
    logging.info("[DP4A_Pascal] Node registered (requires Pascal sm_61 GPU for kernel acceleration)")
