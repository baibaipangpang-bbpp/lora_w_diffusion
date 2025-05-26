import torch
import torch.nn as nn
from lora import LoRALinear, LoRAConv2d


def patch_unet_with_lora(unet, r=4, alpha=1.0, dropout=0.0, enable_linear=True, enable_conv=False, conv_filter=None):
    to_patch = []

    for name, module in unet.named_modules():
        # 1. Patch transformer attention (Linear)
        if enable_linear and hasattr(module, "to_q") and isinstance(module.to_q, nn.Linear):
            to_patch.append((module, "to_q", "linear"))
            to_patch.append((module, "to_k", "linear"))
            to_patch.append((module, "to_v", "linear"))

            if isinstance(module.to_out, nn.Sequential):
                to_patch.append((module.to_out, "0", "linear"))
            elif isinstance(module.to_out, nn.Linear):
                to_patch.append((module, "to_out", "linear"))

        # 2. Patch Conv2d layers (optional, filtered)
        if enable_conv:
            for sub_name, submodule in module.named_children():
                if isinstance(submodule, nn.Conv2d):
                    full_name = f"{name}.{sub_name}" if name else sub_name
                    if conv_filter is None or conv_filter(full_name, submodule):
                        to_patch.append((module, sub_name, "conv"))

    # Perform patching
    for module, attr, kind in to_patch:
        orig = getattr(module, attr)
        if kind == "linear":
            setattr(module, attr, LoRALinear(orig, r, alpha, dropout))
        elif kind == "conv":
            setattr(module, attr, LoRAConv2d(orig, r, alpha, dropout))


def conv_filter(name, layer):
    if "mid_block" in name or "up_blocks" in name:
        k_h, k_w = layer.kernel_size
        return not (k_h == 1 and k_w == 1)
    return False


def save_lora_weights(model, path="lora_weights.pth"):
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            lora_state_dict[name + ".lora_down.weight"] = module.lora_down.weight
            lora_state_dict[name + ".lora_up.weight"] = module.lora_up.weight
    torch.save(lora_state_dict, "lora_weights.pth")

# torch.save(
#     {k: v.cpu() for k, v in pipe.unet.state_dict().items() if "lora" in k},
#     "lora_weights.pth"
# )

def load_lora_weights(model, path="lora_weights.pth"):
    state_dict = torch.load(path)
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            down_key = name + ".lora_down.weight"
            up_key = name + ".lora_up.weight"
            if down_key in state_dict:
                module.lora_down.weight.data.copy_(state_dict[down_key])
            if up_key in state_dict:
                module.lora_up.weight.data.copy_(state_dict[up_key])
