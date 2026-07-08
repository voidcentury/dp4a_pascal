import logging
import math
import torch
import torch.nn.functional as F

from comfy.quant_ops import QuantizedTensor
from comfy.ops import cast_bias_weight, uncast_bias_weight

from . import _DP4A_AVAILABLE, _IS_PASCAL

# Chunk M (token dimension) to cap peak GPU memory.
# Each chunk allocates x_int8[M_chunk, K] + out[M_chunk, N] on device.
# A chunk of 2048 tokens with N=14336 costs ~115 MB — safe for 4 GB cards.
_M_CHUNK_SIZE = 8192


def _can_patch_module(m, name=""):
    if not hasattr(m, "forward_comfy_cast_weights"):
        logging.info("Skipping %s: no forward_comfy_cast_weights", name)
        return False
    if not isinstance(getattr(m, "weight", None), QuantizedTensor):
        logging.info("Skipping %s: weight not QuantizedTensor (%s)", name, type(m.weight).__name__)
        return False
    if getattr(m, "layout_type", None) != "TensorWiseINT8Layout":
        logging.info("Skipping %s: layout=%s", name, getattr(m, "layout_type", None))
        return False
    if getattr(m, "comfy_force_cast_weights", False):
        logging.info("Skipping %s: force_cast=True", name)
        return False
    if len(m.weight_function) > 0 or len(m.bias_function) > 0:
        logging.info("Skipping %s: wf=%d bf=%d", name, len(m.weight_function), len(m.bias_function))
        return False
    K = m.weight.shape[-1] if hasattr(m.weight, "shape") else m.in_features
    if K % 4 != 0:
        logging.info("Skipping %s: K=%d not divisible by 4", name, K)
        return False
    return True


def _make_patched_fwd(original_fwd):
    _hit = False
    def patched(self, input, compute_dtype=None, want_requant=False, weight_only_quant=False):
        in_dtype = input.dtype if isinstance(input, torch.Tensor) else torch.float32

        if not (
            weight_only_quant
            and isinstance(self.weight, QuantizedTensor)
            and in_dtype == torch.float32
            and not getattr(self, "comfy_force_cast_weights", False)
            and len(self.weight_function) == 0
            and len(self.bias_function) == 0
        ):
            logging.debug(
                "Fallback (weight_only_quant=%s, is_qt=%s, in_dtype=%s, "
                "force_cast=%s, wf=%d, bf=%d)",
                weight_only_quant,
                isinstance(self.weight, QuantizedTensor),
                in_dtype,
                getattr(self, "comfy_force_cast_weights", False),
                len(self.weight_function),
                len(self.bias_function),
            )
            return original_fwd(self, input, compute_dtype, want_requant, weight_only_quant)

        weight_qt, bias, offload_stream = cast_bias_weight(
            self,
            input=None,
            dtype=self.weight.dtype,
            device=input.device,
            bias_dtype=in_dtype,
            offloadable=True,
            compute_dtype=compute_dtype,
            want_requant=True,
        )

        if not isinstance(weight_qt, QuantizedTensor):
            raise RuntimeError("cast_bias_weight did not return QuantizedTensor")

        if not hasattr(weight_qt, "_qdata") or weight_qt._qdata is None:
            raise RuntimeError("QuantizedTensor missing _qdata")
        if weight_qt._qdata.dtype != torch.int8:
            logging.debug("Fallback: weight _qdata dtype is %s, expected int8", weight_qt._qdata.dtype)
            raise TypeError("weight is not int8")

        int8_data = weight_qt._qdata
        if not hasattr(weight_qt, "params") or weight_qt.params is None:
            raise RuntimeError("QuantizedTensor missing params")
        scale_w = weight_qt.params.scale
        if scale_w.device != input.device:
            scale_w = scale_w.to(device=input.device, dtype=torch.float32)

        K = int8_data.shape[1]
        if K % 4 != 0:
            raise RuntimeError(f"K={K} not divisible by 4")

        input_shape = input.shape
        if input.ndim == 3:
            input_2d = input.reshape(-1, input_shape[2])
        else:
            input_2d = input

        nonlocal _hit
        if not _hit:
            _hit = True
            logging.info(
                "DP4A kernel: M=%d, N=%d, K=%d, int8.shape=%s",
                input_2d.shape[0], int8_data.shape[0], K, list(int8_data.shape),
            )
        else:
            logging.debug(
                "DP4A kernel: M=%d, N=%d, K=%d, input.shape=%s, int8.shape=%s",
                input_2d.shape[0], int8_data.shape[0], K, input_shape, int8_data.shape,
            )

        import dp4a_ext

        M_total = input_2d.shape[0]
        N = int8_data.shape[0]
        out = torch.empty((M_total, N), dtype=in_dtype, device=input.device)
        for start in range(0, M_total, _M_CHUNK_SIZE):
            end = min(start + _M_CHUNK_SIZE, M_total)
            chunk_in = input_2d[start:end]
            try:
                chunk_out = dp4a_ext.int8_linear(chunk_in, int8_data, scale_w, None)
            except Exception as chunk_e:
                logging.info("DP4A kernel fallback chunk [%d:%d]: %s", start, end, chunk_e)
                w_fp = weight_qt._qdata.float() * weight_qt.params.scale
                if type(chunk_in) is not torch.Tensor:
                    chunk_in = chunk_in.as_subclass(torch.Tensor)
                chunk_out = F.linear(chunk_in, w_fp, None)
            out[start:end] = chunk_out
            del chunk_out

        if bias is not None:
            out = out + bias

        if input.ndim == 3:
            out = out.reshape(input_shape[0], input_shape[1], int8_data.shape[0])

        uncast_bias_weight(self, weight_qt, bias, offload_stream)
        return out

    return patched


def _patch_model(model_nn):
    patched_count = 0
    patched_names = []
    for name, m in model_nn.named_modules():
        if not _can_patch_module(m, name):
            continue
        if hasattr(m, "_dp4a_patched"):
            continue

        original = m.forward_comfy_cast_weights.__func__
        patched = _make_patched_fwd(original)
        import types
        m.forward_comfy_cast_weights = types.MethodType(patched, m)
        m._dp4a_patched = True
        patched_count += 1
        patched_names.append(name)
        logging.info("Patched %s (%s) — layout=%s, shape=%s, K=%d",
                     name, type(m).__qualname__,
                     getattr(m, "layout_type", None),
                     list(m.weight.shape) if hasattr(m.weight, "shape") else "?",
                     m.weight._qdata.shape[1] if hasattr(m.weight, "_qdata") else m.in_features)

    logging.info("DP4A Pascal: patched %d layers: %s", patched_count, patched_names)
    return patched_count


class DP4APascalApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/dp4a"

    def apply(self, model):
        if not _IS_PASCAL:
            logging.warning("No Pascal GPU detected — returning model unmodified")
            return (model,)

        if not _DP4A_AVAILABLE:
            logging.warning("dp4a_ext not loaded — returning model unmodified")
            return (model,)

        m = model.clone()
        count = _patch_model(m.model)
        logging.info("DP4A Pascal: patched %d layers", count)
        if count == 0:
            logging.warning("No TensorWiseINT8 layers found to patch")
        return (m,)
