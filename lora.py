import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        # Save dimensions
        in_features = original.in_features
        out_features = original.out_features
        bias = original.bias is not None

        # Clone the frozen original linear layer
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)
        self.base_linear.load_state_dict(original.state_dict())
        for p in self.base_linear.parameters():
            p.requires_grad = False

        # LoRA adapters
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)

        # Initialization
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    def __init__(self, original: nn.Conv2d, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout2d(dropout)

        # Copy hyperparameters
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.bias = original.bias is not None

        # Frozen original layer
        self.base_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias
        )
        self.base_conv.load_state_dict(original.state_dict())
        for p in self.base_conv.parameters():
            p.requires_grad = False

        # LoRA adapters
        self.lora_down = nn.Conv2d(self.in_channels, r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(r, self.out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base_conv(x) + self.dropout(self.lora_up(self.lora_down(x))) * self.scaling

