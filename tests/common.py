import safetensors.torch
import torch
from pathlib import Path

WEIGHT_FILE = Path(__file__).resolve().parent.parent / "test_mlp_block.safetensors"

def load_mlp_weights(device="cpu"):
    data = safetensors.torch.load_file(str(WEIGHT_FILE))
    return {
        "ffn0_w_int8": data["blocks.5.ffn.0.weight"],       # [14336, 3072] int8
        "ffn0_scale": data["blocks.5.ffn.0.weight_scale"],    # scalar fp32
        "ffn0_bias": data["blocks.5.ffn.0.bias"],             # [14336] fp16
        "ffn2_w_int8": data["blocks.5.ffn.2.weight"],         # [3072, 14336] int8
        "ffn2_scale": data["blocks.5.ffn.2.weight_scale"],    # scalar fp32
        "ffn2_bias": data["blocks.5.ffn.2.bias"],             # [3072] fp16
    }

def dequant_weight(w_int8, scale, device="cuda"):
    return w_int8.to(device).float() * scale.to(device).item()

def baseline_linear(x, w_int8, scale, bias):
    w_fp32 = dequant_weight(w_int8, scale, x.device)
    b = bias.to(x.device).float() if bias is not None else None
    return torch.nn.functional.linear(x, w_fp32, b)
