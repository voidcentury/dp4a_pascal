# DeepWiki Q&A with Code Context for Repository: Comfy-Org/ComfyUI
## Q1
on pascal gpu (cuda < 13), when using int8 tensorwise model, i plan to write a custom node that overrides all linear layers to route through a custom kernel. The kernel directly operates on fp16/fp32 inputs and int8 weights without dequantizing the int8. Will this work if the model has lora nodes when the lora weights are in f16/f32? will there be complications?
### Answer
The user is asking about a technical question regarding:
1. Pascal GPU (CUDA < 13, meaning CUDA compute capability < 8.0 or similar)
2. Int8 tensorwise quantized models in ComfyUI
3. Custom node that overrides linear layers with a custom kernel
4. The kernel operates on fp16/fp32 inputs and int8 weights without dequantizing
5. Whether this will work when LoRA weights are in f16/f32

Let me search for how ComfyUI handles:
1. Int8 quantization / tensorwise models
2. LoRA weight application
3. Linear layer patching/overriding

Now I have a comprehensive understanding of how ComfyUI handles int8 tensorwise quantization and LoRA. Let me synthesize the key findings to answer the user's question.

The user wants to:
1. Write a custom node that overrides all linear layers to route through a custom kernel
2. The kernel operates on fp16/fp32 inputs and int8 weights WITHOUT dequantizing
3. They want to know if this will work when LoRA weights are in f16/f32

Key findings:

1. **int8_tensorwise format**: The `QUANT_ALGOS["int8_tensorwise"]` has `"quantize_input": False`, meaning the input is NOT quantized - only the weight is int8. This is weight-only quantization.

2. **The critical gate in `MixedPrecisionOps.Linear.forward()`**: The `_use_quantized` flag is set to `False` when `len(self.weight_function) > 0 or len(self.bias_function) > 0`. This means when LoRA is active (via `weight_function`), the quantized fast path is completely bypassed.

3. **What happens with LoRA**: When LoRA patches are present, they are applied via `weight_function` callbacks. In `cast_bias_weight()`, when `weight_has_function` is True:
   - The weight is cast to `dtype` (fp16/bf16)
   - If it's a `QuantizedTensor`, it's **dequantized** first
   - Then the LoRA functions are applied to the dequantized weight
   - The result is a plain fp16/bf16 weight

4. **The LoRA application path**: In `patch_weight_to_device()`, the `convert_func` (which is `convert_weight` on `MixedPrecisionOps.Linear`) is called first, which dequantizes the weight. Then `calculate_weight()` applies the LoRA diff to the dequantized weight. The result is then passed to `set_func` (`set_weight`), which re-quantizes it back to int8.

Wait, let me re-read this more carefully. There are two LoRA application modes:

**Mode 1: Weight patching (standard)** - `patch_weight_to_device()`:
- Gets `convert_func` = `convert_weight` (dequantizes QuantizedTensor)
- Gets `set_func` = `set_weight` (re-quantizes back)
- Calls `convert_func` to dequantize → applies LoRA diff → calls `set_func` to re-quantize
- So the weight stored in the module is re-quantized int8 with LoRA baked in

**Mode 2: Low-VRAM / weight_function path** - when `weight_function` is populated:
- `_use_quantized` becomes `False` (the quantized fast path is skipped)
- `cast_bias_weight()` is called, which dequantizes the weight and applies the weight_function (LoRA)
- The result passed to `_forward()` is a plain fp16/bf16 tensor

So the question is: **which mode will be active when the user's custom kernel is called?**

If the user replaces the `forward()` method of the linear layers with their custom kernel:
- In **Mode 1** (weight patching): The weight stored in the module will be a re-quantized `QuantizedTensor` (int8). The custom kernel would receive the int8 weight directly. LoRA is already baked in. This would work.
- In **Mode 2** (weight_function/low-VRAM): The `weight_function` list is non-empty. The existing `MixedPrecisionOps.Linear.forward()` would skip the quantized path and call `cast_bias_weight()` which dequantizes. But if the user has replaced `forward()`, they need to handle this themselves.

The key complication: **When LoRA is active via `weight_function`** (low-VRAM mode), the `_use_quantized` flag is explicitly set to `False`:

```python
_use_quantized = (
    getattr(self, 'layout_type', None) is not None and
    not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
    not getattr(self, 'comfy_force_cast_weights', False) and
    len(self.weight_function) == 0 and len(self.bias_function) == 0  # <-- LoRA disables quantized path
)
```

So if the user's custom node replaces `forward()` but doesn't check `weight_function`, they'll miss the LoRA application entirely. If they do check it, they need to handle the case where the weight is a `QuantizedTensor` and needs to be dequantized before LoRA functions can be applied.

**The two LoRA modes in ComfyUI:**

1. **Standard patching** (full VRAM): LoRA is baked into the weight via `patch_weight_to_device()`. The weight is dequantized, LoRA applied, then re-quantized via `set_weight()`. The module's `self.weight` is a `QuantizedTensor` with LoRA baked in. The custom kernel would see the int8 weight with LoRA already merged.

2. **Low-VRAM / weight_function mode**: LoRA is applied on-the-fly via `weight_function` callbacks. The `weight_function` list is non-empty. The existing code explicitly disables the quantized path in this case. If the user's custom kernel replaces `forward()` without handling `weight_function`, LoRA will be silently ignored.

3. **Bypass LoRA mode** (`load_bypass_lora_for_models`): LoRA is applied as a forward hook that adds `lora_up(lora_down(x))` to the output. This doesn't touch the weight at all, so the custom kernel would compute the base output correctly, and the bypass hook would add the LoRA contribution on top. This mode would work transparently.

So the complications are:
1. In weight_function mode (low-VRAM), the user's custom kernel needs to explicitly check `weight_function` and handle dequantization + LoRA application, or LoRA will be silently dropped.
2. In standard patching mode, the weight is re-quantized after LoRA is baked in, so the custom kernel sees int8 with LoRA merged - this works.
3. In bypass mode, the custom kernel works fine for the base computation.

Let me also note: for Pascal GPUs (CUDA < 13), the `ck.registry.disable("cuda")` is called, meaning the optimized CUDA kernels from `comfy_kitchen` are disabled. So the `TensorWiseINT8Layout` would fall back to dequantize + standard matmul. The user wants to provide their own kernel to replace this.

The user's custom kernel approach: replacing `forward()` on all linear layers. The key question is what `self.weight` looks like when their kernel is called.

Summary of complications:
1. **weight_function (low-VRAM LoRA)**: `self.weight` is still a `QuantizedTensor` (int8), but `weight_function` contains LoRA callbacks. The user's kernel must check for `weight_function` and handle it (dequantize → apply LoRA → either re-quantize or use fp16 weight). If they ignore `weight_function`, LoRA is silently dropped.
2. **Standard patching (full VRAM LoRA)**: `self.weight` is a `QuantizedTensor` (int8) with LoRA already baked in. The custom kernel works correctly.
3. **Bypass LoRA**: Works transparently.

The most important complication is the `weight_function` case. The user needs to check `len(self.weight_function) > 0` and handle it appropriately.


There are real complications here. Let me walk through exactly what happens at each layer of the stack.

## How int8_tensorwise + LoRA interact

### The `_use_quantized` gate

The `MixedPrecisionOps.Linear.forward()` has an explicit guard that **disables the quantized path when LoRA is active via `weight_function`**:

```python
_use_quantized = (
    getattr(self, 'layout_type', None) is not None and
    ...
    len(self.weight_function) == 0 and len(self.bias_function) == 0  # LoRA kills this
)
``` ComfyUI:1242-1247 

Also note: `int8_tensorwise` has `"quantize_input": False`, so it's weight-only quantization — the input is always fp16/bf16. ComfyUI:223-228 

### The three LoRA modes and what your kernel sees

**Mode 1: Standard weight patching (full VRAM)**

`patch_weight_to_device()` calls `convert_func` (which is `convert_weight`, dequantizing the `QuantizedTensor`), applies the LoRA diff in fp32, then calls `set_func` (`set_weight`), which **re-quantizes the result back to int8**: ComfyUI:856-875 ComfyUI:1301-1316 

In this mode, `self.weight` is a `QuantizedTensor` (int8) with LoRA already baked in. Your custom kernel sees int8 weights and works correctly.

**Mode 2: Low-VRAM / `weight_function` path**

LoRA is applied on-the-fly. `self.weight_function` is non-empty. In `cast_bias_weight()`, when `weight_has_function` is true, the weight is **dequantized to fp16/bf16 first**, then the LoRA functions are applied:

```python
if weight_has_function or weight.dtype != dtype:
    weight = weight.to(dtype=dtype)
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()   # <-- dequantized here
    for f in s.weight_function:
        weight = f(weight)             # <-- LoRA applied to fp16 weight
``` ComfyUI:375-380 

**This is the critical complication for your custom node.** If you replace `forward()` without checking `weight_function`, you'll call your int8 kernel with the raw int8 weight and **silently drop the LoRA contribution entirely**. The LoRA functions in `weight_function` are never called.

**Mode 3: Bypass LoRA (`load_bypass_lora_for_models`)**

LoRA is injected as a forward hook that computes `output += lora_up(lora_down(x))` and adds it to the base output. Your kernel computes the base matmul correctly, and the hook adds the LoRA term on top. This mode works transparently with your approach. ComfyUI:125-186 

## What you need to handle

In your custom `forward()`, you must check for `weight_function`:

```python
def forward(self, input):
    if len(self.weight_function) > 0 or len(self.bias_function) > 0:
        # LoRA is active on-the-fly — must dequantize and apply weight_function
        # before calling your kernel, or fall back to standard path
        weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
        # weight is now fp16/bf16 with LoRA applied — can't use your int8 kernel
        x = torch.nn.functional.linear(input, weight, bias)
        uncast_bias_weight(self, weight, bias, offload_stream)
        return x
    
    # Safe to use your int8 kernel — weight is QuantizedTensor, LoRA is baked in or absent
    return your_int8_kernel(input, self.weight, self.bias)
```

This mirrors exactly what `fp8_ops.Linear.forward_comfy_cast_weights` does — it checks `weight_function` before attempting the fast path: ComfyUI:859-871 

## Summary of complications

| LoRA mode | `self.weight` dtype | `weight_function` | Your kernel safe? |
|---|---|---|---|
| Standard patching (full VRAM) | int8 (LoRA baked in) | empty | Yes |
| Low-VRAM `weight_function` | int8 (LoRA NOT baked in) | non-empty | **No — LoRA silently dropped** |
| Bypass injection (hook) | int8 | empty | Yes |

The only dangerous case is the `weight_function` low-VRAM path. You must detect it and fall back to a dequantized path, since the LoRA delta is in fp16/fp32 and cannot be applied to int8 weights without dequantizing first.
### Citations
**File:** comfy/ops.py (L375-380)
```python
    if weight_has_function or weight.dtype != dtype:
        weight = weight.to(dtype=dtype)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        for f in s.weight_function:
            weight = f(weight)
```
**File:** comfy/ops.py (L859-871)
```python
        def forward_comfy_cast_weights(self, input):
            if len(self.weight_function) == 0 and len(self.bias_function) == 0:
                try:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out
                except Exception as e:
                    logging.info("Exception during fp8 op: {}".format(e))

            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x
```
**File:** comfy/ops.py (L1242-1247)
```python
                _use_quantized = (
                    getattr(self, 'layout_type', None) is not None and
                    not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
                    not getattr(self, 'comfy_force_cast_weights', False) and
                    len(self.weight_function) == 0 and len(self.bias_function) == 0
                )
```
**File:** comfy/ops.py (L1301-1316)
```python
            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, 'layout_type', None) is not None:
                    weight = self.weight.requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
                self.weight = torch.nn.Parameter(weight, requires_grad=False)
```
**File:** comfy/quant_ops.py (L223-228)
```python
QUANT_ALGOS["int8_tensorwise"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": "TensorWiseINT8Layout",
    "quantize_input": False,
}
```
**File:** comfy/model_patcher.py (L856-875)
```python
        temp_dtype = comfy.model_management.lora_compute_dtype(device_to) if key in self.patches else None
        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, temp_dtype, copy=True)
        else:
            temp_weight = weight.to(temp_dtype, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key) if key in self.patches else temp_weight
        if set_func is None:
            if key in self.patches:
                out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=comfy.utils.string_to_seed(key))
            if return_weight:
                return out_weight
            elif inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)
        else:
            return set_func(out_weight, inplace_update=inplace_update, seed=comfy.utils.string_to_seed(key), return_weight=return_weight)
```
**File:** comfy/sd.py (L125-186)
```python
def load_bypass_lora_for_models(model, clip, lora, strength_model, strength_clip):
    """
    Load LoRA in bypass mode without modifying base model weights.

    Instead of patching weights, this injects the LoRA computation into the
    forward pass: output = base_forward(x) + lora_path(x)

    Non-adapter patches (bias diff, weight diff, etc.) are applied as regular patches.

    This is useful for training and when model weights are offloaded.
    """
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    logging.debug(f"[BypassLoRA] key_map has {len(key_map)} entries")

    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map)

    logging.debug(f"[BypassLoRA] loaded has {len(loaded)} entries")

    # Separate adapters (for bypass) from other patches (for regular patching)
    bypass_patches = {}  # WeightAdapterBase instances -> bypass mode
    regular_patches = {}  # diff, set, bias patches -> regular weight patching

    for key, patch_data in loaded.items():
        if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
            bypass_patches[key] = patch_data
        else:
            regular_patches[key] = patch_data

    logging.debug(f"[BypassLoRA] {len(bypass_patches)} bypass adapters, {len(regular_patches)} regular patches")

    k = set()
    k1 = set()

    if model is not None:
        new_modelpatcher = model.clone()

        # Apply regular patches (bias diff, weight diff, etc.) via normal patching
        if regular_patches:
            patched_keys = new_modelpatcher.add_patches(regular_patches, strength_model)
            k.update(patched_keys)

        # Apply adapter patches via bypass injection
        manager = comfy.weight_adapter.BypassInjectionManager()
        model_sd_keys = set(new_modelpatcher.model.state_dict().keys())

        for key, adapter in bypass_patches.items():
            if key in model_sd_keys:
                manager.add_adapter(key, adapter, strength=strength_model)
                k.add(key)
            else:
                logging.warning(f"[BypassLoRA] Adapter key not in model state_dict: {key}")

        injections = manager.create_injections(new_modelpatcher.model)

        if manager.get_hook_count() > 0:
            new_modelpatcher.set_injections("bypass_lora", injections)
```
## Q2
i am going to use the bypass lora mode. Give instructions to create the custom node focusing on seamless integration: how to hook, what to override and implement, and how to fallback to default path, return, etc.
### Answer
The user wants detailed instructions for creating a custom node that:
1. Overrides all linear layers to route through a custom int8 kernel
2. Uses bypass LoRA mode (so `weight_function` is always empty)
3. Needs seamless integration with ComfyUI's existing infrastructure

Let me look at the relevant code to understand:
1. How `MixedPrecisionOps.Linear` is structured (the forward method, the quantized path)
2. How `cast_bias_weight` and `uncast_bias_weight` work
3. How `TensorWiseINT8Layout` works
4. How to properly hook into ComfyUI's node system
5. The `fp8_linear` pattern as a reference for how to implement a custom fast path

I need to look at:
- `comfy/ops.py` - the MixedPrecisionOps.Linear class and its forward method
- `comfy/quant_ops.py` - TensorWiseINT8Layout
- How existing custom nodes are structured

I now have enough context to give a comprehensive answer. Let me synthesize everything.

The user wants to create a custom node that:
1. Overrides all linear layers in an int8 tensorwise model to use a custom kernel
2. Uses bypass LoRA mode (so `weight_function` is always empty)
3. Needs seamless integration with ComfyUI

Key facts:
1. The linear layers in an int8 tensorwise model are instances of `MixedPrecisionOps.Linear` (from `comfy/ops.py`)
2. `self.weight` is a `QuantizedTensor` wrapping int8 data with a `TensorWiseINT8Layout`
3. The scale is stored in `self.weight._params.weight_scale` (or similar)
4. In bypass LoRA mode, `weight_function` is always empty, so the quantized path is always taken
5. The `_use_quantized` flag will be True (since `weight_function` is empty, `layout_type` is set, etc.)
6. The `forward_comfy_cast_weights` with `weight_only_quant=True` is the path taken for int8_tensorwise

The approach:
1. Iterate over all linear modules in the model
2. For each `MixedPrecisionOps.Linear` with `layout_type == "TensorWiseINT8Layout"`, monkey-patch the `forward` method
3. In the patched forward, check if the weight is a `QuantizedTensor` with int8 data
4. Extract the raw int8 data and scale, call the custom kernel
5. Fall back to the default path if anything is unexpected

The key things to understand:
- `self.weight` is a `QuantizedTensor` - to get the raw int8 tensor, use `self.weight.data` or `self.weight._data`
- The scale is in `self.weight._params.weight_scale` (for TensorWiseINT8Layout)
- The bias needs to be moved to the right device
- Need to handle 3D input reshaping (the original code does this)
- Need to handle `comfy_cast_weights` flag and `comfy_force_cast_weights`
- Need to handle the offload stream via `cast_bias_weight` / `uncast_bias_weight` for the bias

Let me think about the structure more carefully.

The `MixedPrecisionOps.Linear.forward()` does:
1. Check `_use_quantized` - True when `layout_type` is set, `weight_function` is empty, etc.
2. Since `int8_tensorwise` has `quantize_input: False`, it goes to `weight_only_quant = True`
3. Calls `forward_comfy_cast_weights(input, compute_dtype, want_requant=False, weight_only_quant=True)`
4. In `forward_comfy_cast_weights` with `weight_only_quant=True`:
   - Calls `cast_bias_weight(self, input=None, dtype=self.weight.dtype, device=input.device, bias_dtype=input.dtype, offloadable=True, compute_dtype=compute_dtype, want_requant=True)`
   - Then `weight = weight.to(dtype=input.dtype)` - this dequantizes!
   - Then calls `self._forward(input, weight, bias)` = `F.linear(input, weight, bias)`

Wait, so the default path for int8_tensorwise DOES dequantize the weight before the matmul? Let me re-read...

```python
if weight_only_quant:
    weight, bias, offload_stream = cast_bias_weight(
        self,
        input=None,
        dtype=self.weight.dtype,  # keeps it as int8 dtype
        device=input.device,
        bias_dtype=input.dtype,
        offloadable=True,
        compute_dtype=compute_dtype,
        want_requant=True,
    )
    weight = weight.to(dtype=input.dtype)  # THIS converts int8 -> fp16/bf16 (dequantizes)
```

So `cast_bias_weight` with `want_requant=True` returns the weight as a `QuantizedTensor` (still int8), then `weight.to(dtype=input.dtype)` dequantizes it to fp16/bf16. Then `F.linear` is called with fp16 input and fp16 weight.

Wait, but `QuantizedTensor.to(dtype)` - does this dequantize? Let me think... `QuantizedTensor` is a subclass of `torch.Tensor` (or wraps one). When you call `.to(dtype=fp16)` on a `QuantizedTensor`, it likely dequantizes.

So the default path for int8_tensorwise on Pascal (where CUDA kernels are disabled) is:
1. Get the QuantizedTensor weight (int8 + scale)
2. Call `.to(dtype=input.dtype)` which dequantizes to fp16/bf16
3. Call `F.linear(input_fp16, weight_fp16, bias_fp16)`

The user wants to replace step 2+3 with their custom kernel that operates directly on int8 weights without dequantizing.

So the custom forward needs to:
1. Get the raw int8 data and scale from `self.weight` (a QuantizedTensor)
2. Move bias to the right device
3. Call the custom kernel with fp16 input, int8 weight, scale
4. Handle the offload stream for the bias

For the patching approach, the cleanest way is to monkey-patch `forward` on each qualifying module. The `nodes_train.py` shows the pattern: save `org_forward`, replace `m.forward`.

Let me now think about what the custom forward should look like:

```python
def make_int8_forward(module, original_forward):
    def custom_forward(input, *args, **kwargs):
        from comfy.quant_ops import QuantizedTensor
        from comfy.ops import cast_bias_weight, uncast_bias_weight
        
        # Check if we can use the custom kernel
        # In bypass LoRA mode, weight_function is always empty
        # But we should still check for safety
        can_use_custom = (
            isinstance(module.weight, QuantizedTensor) and
            getattr(module, 'layout_type', None) == 'TensorWiseINT8Layout' and
            not getattr(module, 'comfy_force_cast_weights', False) and
            len(module.weight_function) == 0 and
            len(module.bias_function) == 0 and
            not isinstance(input, QuantizedTensor)
        )
        
        if not can_use_custom:
            return original_forward(input, *args, **kwargs)
        
        # Handle 3D input
        input_shape = input.shape
        reshaped_3d = input.ndim == 3
        if reshaped_3d:
            input = input.reshape(-1, input_shape[2])
        
        # Get int8 weight data and scale
        weight_qt = module.weight  # QuantizedTensor
        int8_data = weight_qt.data  # raw int8 tensor (or weight_qt._data)
        scale = weight_qt._params.weight_scale  # per-tensor scale
        
        # Move to correct device
        int8_data = int8_data.to(device=input.device)
        scale = scale.to(device=input.device)
        
        # Handle bias
        bias = None
        offload_stream = None
        if module.bias is not None:
            # Use cast_bias_weight just for the bias, or manually move it
            bias = module.bias.to(device=input.device, dtype=input.dtype)
        
        # Call custom kernel
        output = your_int8_kernel(input, int8_data, scale, bias)
        
        if reshaped_3d:
            output = output.reshape(input_shape[0], input_shape[1], module.weight.shape[0])
        
        return output
    
    return custom_forward
```

But I'm not sure about the exact API for accessing the raw data from a `QuantizedTensor`. The `QuantizedTensor` is from `comfy_kitchen`. I don't have access to its internals. It's likely `weight_qt.data` gives the raw int8 tensor, and the scale is in `weight_qt._params.weight_scale` or similar.

Actually, looking at the code more carefully:

In `quant_ops.py`:
```python
TensorWiseINT8Layout = _CKTensorWiseINT8Layout
```

And `QuantizedTensor` is from `comfy_kitchen`. The `QuantizedTensor` is constructed as:
```python
QuantizedTensor(w, "TensorWiseINT8Layout", params)
```

Where `w` is the int8 data tensor. So `weight_qt.data` should give the underlying int8 tensor (since `QuantizedTensor` is a `torch.Tensor` subclass, `.data` gives the raw storage).

For the scale, looking at how it's used in `_quantized_weight_state_dict`:
```python
if module.quant_format == "int8_tensorwise" and getattr(params, "convrot", False):
```

And in `_load_quantized_module`, the scale is loaded as `weight_scale`. The params object has a `weight_scale` attribute.

So `weight_qt._params.weight_scale` should give the scale tensor.

Actually, I should be more careful here. The `QuantizedTensor` from `comfy_kitchen` - I don't know its exact API. But based on how it's constructed and used, I can make reasonable assumptions.

Let me also think about the patching mechanism. The user wants to create a custom node. The node should:
1. Take a model as input
2. Patch all linear layers
3. Return the patched model

The key question is: should the patching be done on the `ModelPatcher` level or directly on the model's modules?

Looking at how `nodes_train.py` does it with `patch()`:
```python
def patch(m, offloading=False):
    org_forward = m.forward
    ...
    m.org_forward = org_forward
    m.forward = checkpointing_fwd
```

This directly patches the module's `forward` method. This is the simplest approach.

But there's a subtlety: the model is loaded/unloaded from VRAM by ComfyUI's memory management. When the model is loaded, `patch_weight_to_device` is called. The module's `forward` method is on the Python object, not the weight, so it persists across load/unload cycles.

However, there's another subtlety: `model.clone()` creates a new `ModelPatcher` but shares the same underlying model object. So if you patch the modules directly, the patch affects all clones. This might be fine for the user's use case.

The cleanest approach for a custom node:
1. Clone the model patcher: `m = model.clone()`
2. Iterate over all linear modules in `m.model`
3. For each qualifying module, monkey-patch `forward`
4. Return the patched model patcher

For cleanup/unpatching, the user should also implement an unpatch mechanism, but for a simple custom node this might not be necessary.

Let me now write the actual instructions.

Key things to cover:
1. Node structure (inputs/outputs)
2. How to iterate over modules
3. What to check before patching
4. The patched forward function structure
5. How to access int8 data and scale from QuantizedTensor
6. How to handle bias (use cast_bias_weight for the bias, or just move it manually)
7. How to handle 3D input reshaping
8. Fallback conditions
9. The `comfy_cast_weights` flag
10. Cleanup/unpatching

One important thing: the `MixedPrecisionOps.Linear` has `comfy_cast_weights = True` (it inherits from `manual_cast` which sets this). So the `forward` method always goes through `forward_comfy_cast_weights`. But the user is replacing `forward` entirely, so this doesn't matter.

Wait, let me check. `MixedPrecisionOps` inherits from `manual_cast`. Let me look at what `manual_cast` is.

From the code I've seen, `manual_cast` is a class in `comfy/ops.py`. The `disable_weight_init.Linear.forward` checks `self.comfy_cast_weights`. The `MixedPrecisionOps.Linear` overrides `forward` directly (lines 1234-1299), so it doesn't use the `comfy_cast_weights` check from `disable_weight_init.Linear`.

Actually, looking at `MixedPrecisionOps.Linear.forward` at line 1234, it directly implements the full forward logic without checking `comfy_cast_weights`. So the user just needs to replace this `forward` method.

Now, about accessing the QuantizedTensor internals. Since `QuantizedTensor` is from `comfy_kitchen` and is a `torch.Tensor` subclass:
- The raw data is accessible via `.data` (standard PyTorch)
- The params (including scale) are stored in `._params` attribute

For `TensorWiseINT8Layout`, the params have a `weight_scale` attribute (a scalar float32 tensor).

Let me also think about the bias handling. The simplest approach is to use `cast_bias_weight` just for the bias, but that's complex. Alternatively, just do `module.bias.to(device=input.device, dtype=input.dtype)` manually. But this doesn't handle the offload stream optimization.

For simplicity, I'll recommend using `cast_bias_weight` with `offloadable=True` and `uncast_bias_weight` for proper async offload support, as the comment in the code says:
```
# NOTE: offloadable=False is a legacy mode and if you are a custom node author reading this please pass
# offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias.
```

But `cast_bias_weight` also handles the weight. We can call it with `want_requant=True` to get the QuantizedTensor back (not dequantized), then extract the int8 data ourselves.

Actually, looking at the `weight_only_quant` path more carefully:

```python
weight_only_quant = _use_quantized and not quantize_input and isinstance(self.weight, QuantizedTensor)
output = self.forward_comfy_cast_weights(
    input,
    compute_dtype,
    want_requant=isinstance(input, QuantizedTensor),
    weight_only_quant=weight_only_quant,
)
```

For int8_tensorwise, `quantize_input=False`, so `weight_only_quant=True`. Then in `forward_comfy_cast_weights`:

```python
if weight_only_quant:
    weight, bias, offload_stream = cast_bias_weight(
        self,
        input=None,
        dtype=self.weight.dtype,  # int8
        device=input.device,
        bias_dtype=input.dtype,
        offloadable=True,
        compute_dtype=compute_dtype,
        want_requant=True,
    )
    weight = weight.to(dtype=input.dtype)  # dequantize
```

So `cast_bias_weight` with `want_requant=True` returns the weight as a `QuantizedTensor` (still int8). Then `.to(dtype=input.dtype)` dequantizes it.

For the custom kernel, we want to intercept after `cast_bias_weight` returns the `QuantizedTensor` but before the `.to(dtype=input.dtype)` dequantization.

So the cleanest approach is to override `forward_comfy_cast_weights` instead of `forward`, since `forward` has a lot of logic (3D reshaping, training path, etc.) that we want to keep.

Actually, looking at the code again:

The `forward` method at line 1234 handles:
1. `run_every_op()` - must be called
2. 3D reshaping
3. `_use_quantized` check
4. Training path (if `input.requires_grad`)
5. Inference path with `weight_only_quant`

The `forward_comfy_cast_weights` at line 1203 handles:
1. `weight_only_quant` branch: gets QuantizedTensor, then dequantizes
2. Regular branch: gets fp16 weight (with LoRA applied if weight_function is set)

The best override point is `forward_comfy_cast_weights`, specifically the `weight_only_quant` branch. But since we're in bypass LoRA mode (weight_function is empty), we can also override `_forward` which is called at line 1230.

Wait, `_forward` is called with `(input, weight, bias)` where weight is already dequantized fp16. So overriding `_forward` won't help - we'd still get fp16 weight.

The best approach is to override `forward_comfy_cast_weights` to intercept the `weight_only_quant` path before dequantization.

OR, even simpler: override `forward` entirely, keeping the 3D reshape logic and the `_use_quantized` check, but replacing the `weight_only_quant` path with our custom kernel call.

Let me think about what's the cleanest approach for a custom node author:

**Option A: Override `forward` entirely**
- Copy the full `forward` logic
- Replace the `weight_only_quant` path with custom kernel
- Fragile: breaks if ComfyUI updates `forward`

**Option B: Override `forward_comfy_cast_weights`**
- Only override the method that handles the weight_only_quant path
- The `forward` method still handles 3D reshaping, `_use_quantized` check, etc.
- More robust

**Option C: Override `_forward`**
- Called with already-dequantized weight - doesn't help

**Option D: Monkey-patch `forward` to call original but intercept at the right point**
- Complex

I think Option B is the cleanest. Override `forward_comfy_cast_weights` to:
1. Check if `weight_only_quant` conditions are met
2. If yes, extract int8 data and scale, call custom kernel
3. Otherwise, call the original `forward_comfy_cast_weights`

But wait, `forward_comfy_cast_weights` doesn't receive `weight_only_quant` as a parameter - it's passed from `forward`. Let me re-read...

```python
def forward_comfy_cast_weights(
    self,
    input,
    compute_dtype=None,
    want_requant=False,
    weight_only_quant=False,
):
```

Yes, `weight_only_quant` is a parameter. So we can override `forward_comfy_cast_weights` and check `weight_only_quant`.

Here's the plan:

```python
def patched_forward_comfy_cast_weights(self, input, compute_dtype=None, want_requant=False, weight_only_quant=False):
    from comfy.quant_ops import QuantizedTensor
    from comfy.ops import cast_bias_weight, uncast_bias_weight
    
    if weight_only_quant and isinstance(self.weight, QuantizedTensor):
        # Get the QuantizedTensor (still int8) and bias
        weight_qt, bias, offload_stream = cast_bias_weight(
            self,
            input=None,
            dtype=self.weight.dtype,  # keep as int8
            device=input.device,
            bias_dtype=input.dtype,
            offloadable=True,
            compute_dtype=compute_dtype,
            want_requant=True,
        )
        
        try:
            # Extract raw int8 data and scale
            int8_data = weight_qt.data  # raw int8 tensor
            scale = weight_qt._params.weight_scale.to(device=input.device)
            
            # Call custom kernel
            output = your_custom_int8_kernel(input, int8_data, scale, bias)
            uncast_bias_weight(self, weight_qt, bias, offload_stream)
            return output
        except Exception as e:
            logging.warning(f"Custom int8 kernel failed: {e}, falling back to default")
            # Fall through to default path
            weight = weight_qt.to(dtype=input.dtype)  # dequantize
            x = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight_qt, bias, offload_stream)
            return x
    
    # Fall back to original for all other cases
    return original_forward_comfy_cast_weights(self, input, compute_dtype, want_requant, weight_only_quant)
```

Hmm, but there's a problem: `cast_bias_weight` with `want_requant=True` - what does this actually do? Let me re-read the `cast_bias_weight` function...

Looking at lines 280-382, `cast_bias_weight` has a `want_requant` parameter. In the vbar path (lines 316-334), it calls `resolve_cast_module_with_vbar(s, dtype, device, bias_dtype, compute_dtype, want_requant)`. In the regular path (lines 337-382), it doesn't seem to use `want_requant` directly - it just moves the weight to the device and applies `weight_function`.

Actually, looking at lines 375-380:
```python
if weight_has_function or weight.dtype != dtype:
    weight = weight.to(dtype=dtype)
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    for f in s.weight_function:
        weight = f(weight)
```

So if `weight.dtype == dtype` (both int8), it won't enter this branch and the weight stays as a `QuantizedTensor`. That's what `want_requant=True` achieves - by passing `dtype=self.weight.dtype` (int8), the weight stays as a QuantizedTensor.

Wait, but `want_requant` is passed to `cast_bias_weight` but I don't see it used in the regular path (lines 337-382). Let me look more carefully...

Actually, looking at the code again, in the `weight_only_quant` path:
```python
weight, bias, offload_stream = cast_bias_weight(
    self,
    input=None,
    dtype=self.weight.dtype,  # int8 dtype
    device=input.device,
    bias_dtype=input.dtype,
    offloadable=True,
    compute_dtype=compute_dtype,
    want_requant=True,
)
weight = weight.to(dtype=input.dtype)  # dequantize
```

The key is `dtype=self.weight.dtype` (int8). In `cast_bias_weight`, line 375:
```python
if weight_has_function or weight.dtype != dtype:
```

Since `weight.dtype == dtype` (both int8) and `weight_function` is empty (bypass LoRA mode), this condition is False, so the weight is NOT dequantized. The returned `weight` is still a `QuantizedTensor` with int8 data.

Then `weight.to(dtype=input.dtype)` dequantizes it.

So for our custom kernel, we call `cast_bias_weight` the same way (with `dtype=self.weight.dtype`), get back the `QuantizedTensor`, extract the int8 data and scale, and call our kernel instead of dequantizing.

Now, how to access the int8 data from a `QuantizedTensor`? Since `QuantizedTensor` is from `comfy_kitchen` and is a `torch.Tensor` subclass, the raw data is in `.data`. The params (including scale) are in `._params`.

For `TensorWiseINT8Layout`, the params have `weight_scale` (a scalar float32 tensor).

I'm not 100% sure about the exact attribute names since I don't have access to `comfy_kitchen` source. But based on how it's used in the codebase:
- `weight_qt.data` - the raw int8 tensor
- `weight_qt._params.weight_scale` - the scale

Actually, I should note this uncertainty to the user.

Let me also think about the node structure. The user wants a custom node that:
1. Takes a model as input
2. Patches all linear layers
3. Returns the patched model

The node should use `model.clone()` to avoid modifying the original model patcher.

But wait - the modules are shared between the original and cloned model patcher (they share the same underlying `model` object). So patching modules directly would affect all clones. This is a problem.

Actually, looking at how `nodes_train.py` does it - it patches modules directly on `m.model` after cloning. The clone creates a new `ModelPatcher` but the underlying `model` (the `torch.nn.Module`) is shared. So patching `forward` on modules would affect all model patchers that share the same model.

For a custom node, this might be acceptable if the user is always using this node in their workflow. But it's not clean.

A cleaner approach: use `model.add_object_patch` or similar mechanism. But I'm not sure if ComfyUI has a mechanism for patching module forward methods through the model patcher.

Actually, looking at the `nodes_train.py` approach more carefully - it patches `m.forward` (the module's forward method) directly. This is a common pattern in ComfyUI custom nodes. The key is to save the original forward and restore it when done.

But for a custom node that's applied once and the model is used for inference, the patching persists for the lifetime of the model in memory. This is fine for the user's use case.

Let me now write the actual instructions. I'll focus on:
1. The node structure
2. The patching mechanism (iterate modules, check type, monkey-patch forward_comfy_cast_weights)
3. The patched forward implementation
4. Fallback conditions
5. Important caveats

One more thing: the `run_every_op()` call at the start of `forward`. This is important - it's called in the original `forward` method. Since we're overriding `forward_comfy_cast_weights` (not `forward`), `run_every_op()` is still called by the original `forward`. Good.

Also, the 3D reshaping is handled in `forward`, not `forward_comfy_cast_weights`. So by overriding `forward_comfy_cast_weights`, we don't need to worry about 3D reshaping - it's already handled.

Wait, but looking at the `forward` code again:

```python
def forward(self, input, *args, **kwargs):
    run_every_op()
    input_shape = input.shape
    reshaped_3d = False
    compute_dtype = input.dtype
    
    _use_quantized = (...)
    quantize_input = QUANT_ALGOS.get(...).get("quantize_input", True)
    
    # Training path...
    
    weight_only_quant = _use_quantized and not quantize_input and isinstance(self.weight, QuantizedTensor)
    output = self.forward_comfy_cast_weights(
        input,
        compute_dtype,
        want_requant=isinstance(input, QuantizedTensor),
        weight_only_quant=weight_only_quant,
    )
    
    # Reshape output back to 3D if input was 3D
    if reshaped_3d:
        output = output.reshape((input_shape[0], input_shape[1], self.weight.shape[0]))
    
    return output
```

Wait, `reshaped_3d` is set to True only in the `quantize_input=True` path (lines 1276-1285). For int8_tensorwise, `quantize_input=False`, so `reshaped_3d` stays False. The 3D reshaping in `forward` only applies to the input quantization path, not the weight_only_quant path.

So for int8_tensorwise, the input is passed as-is (potentially 3D) to `forward_comfy_cast_weights`, and the output is returned as-is. The 3D handling is not done in `forward` for this case.

This means our custom kernel in `forward_comfy_cast_weights` needs to handle 3D inputs itself. Let me check...

Actually, `F.linear` handles 3D inputs natively (it treats the last dimension as the feature dimension). So if our custom kernel also handles 3D inputs, we're fine. If not, we need to reshape.

For a typical int8 matmul kernel, it likely expects 2D inputs. So we should reshape 3D to 2D before calling the kernel and reshape back after.

OK, I think I have enough to write a comprehensive answer. Let me structure it clearly.

Key points:
1. **What to override**: `forward_comfy_cast_weights` on `MixedPrecisionOps.Linear` instances
2. **How to identify target modules**: check `isinstance(m, torch.nn.Module)` and `getattr(m, 'layout_type', None) == 'TensorWiseINT8Layout'`
3. **The patched forward**: check `weight_only_quant`, call `cast_bias_weight` to get QuantizedTensor, extract int8 data + scale, call custom kernel, call `uncast_bias_weight`
4. **Fallback**: if anything fails or conditions aren't met, call original `forward_comfy_cast_weights`
5. **Node structure**: clone model, iterate modules, patch, return

Let me also note the uncertainty about `QuantizedTensor` internals (`.data` and `._params.weight_scale`).

Actually, I realize I should also mention that the user needs to handle the case where `weight_qt` returned from `cast_bias_weight` might not be a `QuantizedTensor` in some edge cases (e.g., if the weight was already moved to a different device and dequantized). So always check `isinstance(weight_qt, QuantizedTensor)` before extracting int8 data.

Let me write the final answer now.


Here's a complete guide for creating the custom node.

## 1. What to override and why

Override `forward_comfy_cast_weights` on each qualifying module, **not** `forward`. The outer `forward` handles `run_every_op()`, the `_use_quantized` gate, and passes `weight_only_quant=True` down to `forward_comfy_cast_weights` for int8_tensorwise. By overriding only `forward_comfy_cast_weights`, you inherit all of that logic for free. ComfyUI:1203-1232 

The `weight_only_quant=True` branch is the exact path taken for int8_tensorwise (since `"quantize_input": False`). It calls `cast_bias_weight` with `dtype=self.weight.dtype` (int8), which returns the weight **still as a `QuantizedTensor`** because `weight.dtype == dtype` prevents the dequantize branch from firing. Then it immediately dequantizes with `.to(dtype=input.dtype)`. Your kernel intercepts between those two steps. ComfyUI:1210-1231 ComfyUI:1375-1380 

## 2. How to identify target modules

```python
from comfy.quant_ops import QuantizedTensor

def is_int8_tensorwise_linear(m):
    return (
        getattr(m, 'layout_type', None) == 'TensorWiseINT8Layout' and
        isinstance(getattr(m, 'weight', None), QuantizedTensor)
    )
```

## 3. The patched `forward_comfy_cast_weights`

```python
import logging
import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor


def make_patched_fwd(original_fwd):
    def patched_forward_comfy_cast_weights(
        self, input, compute_dtype=None, want_requant=False, weight_only_quant=False
    ):
        # Only intercept the weight-only-quant path for int8 tensorwise.
        # In bypass LoRA mode weight_function is always empty, so _use_quantized
        # is True and weight_only_quant will be True here.
        if not (
            weight_only_quant and
            isinstance(self.weight, QuantizedTensor) and
            not getattr(self, 'comfy_force_cast_weights', False) and
            len(self.weight_function) == 0 and
            len(self.bias_function) == 0
        ):
            return original_fwd(self, input, compute_dtype, want_requant, weight_only_quant)

        # Ask cast_bias_weight to keep the weight as QuantizedTensor (int8).
        # Passing dtype=self.weight.dtype (int8) prevents the dequantize branch.
        weight_qt, bias, offload_stream = cast_bias_weight(
            self,
            input=None,
            dtype=self.weight.dtype,       # keep int8
            device=input.device,
            bias_dtype=input.dtype,
            offloadable=True,
            compute_dtype=compute_dtype,
            want_requant=True,
        )

        try:
            if not isinstance(weight_qt, QuantizedTensor):
                raise RuntimeError("cast_bias_weight returned non-QuantizedTensor")

            # Extract raw int8 data and per-tensor scale.
            # QuantizedTensor is a torch.Tensor subclass; .data is the raw storage.
            int8_data = weight_qt.data                          # torch.int8, shape [out, in]
            scale = weight_qt._params.weight_scale              # float32 scalar tensor
            scale = scale.to(device=input.device, dtype=torch.float32)

            # Handle 3D input: most int8 kernels expect 2D.
            input_shape = input.shape
            if input.ndim == 3:
                input_2d = input.reshape(-1, input_shape[2])
            else:
                input_2d = input

            # ---- call your kernel here ----
            output = your_int8_kernel(input_2d, int8_data, scale, bias)
            # --------------------------------

            if input.ndim == 3:
                output = output.reshape(input_shape[0], input_shape[1], int8_data.shape[0])

            uncast_bias_weight(self, weight_qt, bias, offload_stream)
            return output

        except Exception as e:
            logging.warning(f"[Int8KernelNode] custom kernel failed ({e}), falling back")
            # Fallback: dequantize and use standard F.linear
            weight_fp = weight_qt.to(dtype=input.dtype)
            x = F.linear(input, weight_fp, bias)
            uncast_bias_weight(self, weight_qt, bias, offload_stream)
            return x

    return patched_forward_comfy_cast_weights
```

The `cast_bias_weight` / `uncast_bias_weight` pair is required — it handles async weight offloading. The comment in the source explicitly asks custom node authors to use it: ComfyUI:281-284 

## 4. The node itself

```python
import comfy.model_patcher
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override


def _patch_model(model_nn_module):
    """Walk all modules and patch qualifying Linear layers."""
    for name, m in model_nn_module.named_modules():
        if is_int8_tensorwise_linear(m):
            if hasattr(m, '_int8_kernel_patched'):
                continue  # already patched
            original = m.forward_comfy_cast_weights.__func__  # unbound method
            patched = make_patched_fwd(original)
            # Bind as instance method so `self` is passed correctly
            import types
            m.forward_comfy_cast_weights = types.MethodType(
                lambda self, *a, _p=patched, **kw: _p(self, *a, **kw), m
            )
            m._int8_kernel_patched = True


def _unpatch_model(model_nn_module):
    for name, m in model_nn_module.named_modules():
        if getattr(m, '_int8_kernel_patched', False):
            # Restore by deleting the instance-level override;
            # the class method takes over again.
            if hasattr(m, 'forward_comfy_cast_weights'):
                del m.forward_comfy_cast_weights
            del m._int8_kernel_patched


class ApplyInt8KernelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ApplyInt8Kernel",
            category="model/patch",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()
        _patch_model(m.model)
        return io.NodeOutput(m)
```

The `model.clone()` creates a new `ModelPatcher` but shares the underlying `torch.nn.Module`. Patching instance methods (not class methods) means the patch is on the object, not the class, so it won't affect other model types loaded simultaneously. ComfyUI:512-536 

## 5. Important caveats

**`QuantizedTensor` internals**: `weight_qt.data` and `weight_qt._params.weight_scale` are based on how `comfy_kitchen` constructs `QuantizedTensor` objects. These are not part of a public API. If `comfy_kitchen` changes its internal layout, you'll need to update. Wrap the extraction in a try/except and fall back gracefully. ComfyUI:181-192 

**Shared model object**: Because `model.clone()` shares the underlying `nn.Module`, the patch is visible to all `ModelPatcher` clones of the same model. If you need isolation, you'd need to deep-copy the model, which is expensive.

**`comfy_force_cast_weights`**: This flag is set by the low-VRAM path and forces dequantization. The guard `not getattr(self, 'comfy_force_cast_weights', False)` in the intercept condition ensures you fall back correctly if it's ever set. ComfyUI:1242-1247 

**Bypass LoRA hook fires after your kernel returns**: The `BypassForwardHook` is a PyTorch forward hook registered on the module. It calls `org_forward(x)` (which is your patched forward), then adds `lora_up(lora_down(x))` to the result. You don't need to do anything special — it's transparent. ComfyUI:121-153
### Citations
**File:** comfy/ops.py (L281-284)
```python
def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False, compute_dtype=None, want_requant=False):
    # NOTE: offloadable=False is a legacy mode and if you are a custom node author reading this please pass
    # offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias. This
    # will add async-offload support to your cast and improve performance.
```
**File:** comfy/ops.py (L1203-1232)
```python
            def forward_comfy_cast_weights(
                self,
                input,
                compute_dtype=None,
                want_requant=False,
                weight_only_quant=False,
            ):
                if weight_only_quant:
                    weight, bias, offload_stream = cast_bias_weight(
                        self,
                        input=None,
                        dtype=self.weight.dtype,
                        device=input.device,
                        bias_dtype=input.dtype,
                        offloadable=True,
                        compute_dtype=compute_dtype,
                        want_requant=True,
                    )
                    weight = weight.to(dtype=input.dtype)
                else:
                    weight, bias, offload_stream = cast_bias_weight(
                        self,
                        input,
                        offloadable=True,
                        compute_dtype=compute_dtype,
                        want_requant=want_requant,
                    )
                x = self._forward(input, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return x
```
**File:** comfy/ops.py (L1242-1247)
```python
                _use_quantized = (
                    getattr(self, 'layout_type', None) is not None and
                    not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
                    not getattr(self, 'comfy_force_cast_weights', False) and
                    len(self.weight_function) == 0 and len(self.bias_function) == 0
                )
```
**File:** comfy/ops.py (L1375-1380)
```python

            @contextlib.contextmanager
            def bank_resident(self, input):
                """Cast the whole bank once; expert_linear inside reuses the cast.
                Not re-entrant — do not nest calls on the same instance.
                """
```
**File:** comfy_extras/nodes_train.py (L512-536)
```python
def patch(m, offloading=False):
    if not hasattr(m, "forward"):
        return
    org_forward = m.forward

    # Branch 1: Linear/Conv* -> offload-compatible checkpoint (single input/output)
    if offloading and isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        def checkpointing_fwd(x):
            return OffloadCheckpointFunction.apply(x, org_forward)
    # Branch 2: Others -> standard checkpoint
    else:
        def fwd(args, kwargs):
            return org_forward(*args, **kwargs)

        def checkpointing_fwd(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(fwd, args, kwargs, use_reentrant=False)

    m.org_forward = org_forward
    m.forward = checkpointing_fwd


def unpatch(m):
    if hasattr(m, "org_forward"):
        m.forward = m.org_forward
        del m.org_forward
```
**File:** comfy/quant_ops.py (L181-192)
```python
TensorWiseINT8Layout = _CKTensorWiseINT8Layout


# ==============================================================================
# Registry
# ==============================================================================

register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorWiseINT8Layout", _CKTensorWiseINT8Layout)
```
**File:** comfy/weight_adapter/base.py (L121-153)
```python
    def bypass_forward(
        self,
        org_forward: Callable,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full bypass forward: g(f(x) + h(x, f(x)))

        Note:
            This method does NOT take org_weight/org_bias parameters. Bypass mode
            is designed for quantized models where weights may not be accessible.
            The original forward function handles weight access internally.

        Args:
            org_forward: Original module forward function
            x: Input tensor
            *args, **kwargs: Additional arguments for org_forward

        Returns:
            Output with adapter applied in bypass mode

        Reference: LyCORIS LoConModule.bypass_forward
        """
        # Base forward: f(x)
        base_out = org_forward(x, *args, **kwargs)

        # Additive component: h(x, base_out) - base_out provided for shape reference
        h_out = self.h(x, base_out)

        # Output transformation: g(base + h)
        return self.g(base_out + h_out)
```
