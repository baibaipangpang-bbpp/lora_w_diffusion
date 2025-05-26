import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.original = original
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_dim = original.in_features
        out_dim = original.out_features

        self.lora_down = nn.Linear(in_dim, r, bias=False)
        self.lora_up = nn.Linear(r, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        for p in self.original.parameters():
            p.requires_grad = False
    # forward method 
    def forward(self, x):
        # print(f"x device: {x.device}, lora_down weight device: {self.lora_down.weight.device}")
        return self.original(x) + self.dropout(self.lora_up(self.lora_down(x))) * self.scaling
