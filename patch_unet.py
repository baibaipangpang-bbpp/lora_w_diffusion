# patch_unet.py
from lora import LoRALinear
import torch.nn as nn

def patch_unet_with_lora(unet, r=4, alpha=1.0, dropout=0.0):
    for module in unet.modules():
        if hasattr(module, "to_q") and isinstance(module.to_q, nn.Linear):
            module.to_q = LoRALinear(module.to_q, r, alpha, dropout)
            module.to_k = LoRALinear(module.to_k, r, alpha, dropout)
            module.to_v = LoRALinear(module.to_v, r, alpha, dropout)

            if isinstance(module.to_out, nn.Sequential):
                module.to_out[0] = LoRALinear(module.to_out[0], r, alpha, dropout)
            elif isinstance(module.to_out, nn.Linear):
                module.to_out = LoRALinear(module.to_out, r, alpha, dropout)
