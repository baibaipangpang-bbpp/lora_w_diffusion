import torch
import torch.nn as nn
from lora import LoRALinear, LoRAConv2d
import matplotlib.pyplot as plt
import numpy as np


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
    torch.save(lora_state_dict, path)
    print(f"model written to {path}")

# torch.save(
#     {k: v.cpu() for k, v in pipe.unet.state_dict().items() if "lora" in k},
#     "lora_weights.pth"
# )

def load_lora_weights(model, path="lora_weights.pth"):
    print(f"model to be loaded from {path}")
    state_dict = torch.load(path)
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            down_key = name + ".lora_down.weight"
            up_key = name + ".lora_up.weight"
            if down_key in state_dict:
                module.lora_down.weight.data.copy_(state_dict[down_key])
            if up_key in state_dict:
                module.lora_up.weight.data.copy_(state_dict[up_key])


def get_activation(name):
    """Creates a hook function to capture a layer's output."""
    def hook(model, input, output):
        # Store the output tensor. Detach and move to CPU.
        # We might get multiple outputs (e.g. from Transformer blocks)
        # For Conv2d, it's usually one tensor.
        current_output = output[0] if isinstance(output, tuple) else output
        activations[name] = current_output.detach().cpu()
    return hook


hooks = []
activations = {}

def register_hooks(model, layer_names):
    """Finds layers by name and registers hooks."""
    global hooks
    for name, module in model.named_modules():
        if name in layer_names:
            print(f"Registering hook for: {name}")
            handle = module.register_forward_hook(get_activation(name))
            hooks.append(handle)

def remove_all_hooks():
    """Removes all registered hooks."""
    global hooks
    for handle in hooks:
        handle.remove()
    hooks = []
    print("All hooks removed.")


def plot_activations(layer_name, num_cols=16, scale=1.5, batch_index=0): # Added batch_index
    """Plots the captured activations for a given layer."""
    if layer_name not in activations:
        print(f"No activations found for {layer_name}")
        return

    act = activations[layer_name]
    print(f"Visualizing {layer_name} - Original Shape: {act.shape}")

    # Check batch size and select the specified index
    if act.ndim == 4 and act.shape[0] > 1:
        print(f"Batch size is {act.shape[0]}, selecting index {batch_index}.")
        act = act[batch_index] # Select one image from the batch
    elif act.ndim == 4 and act.shape[0] == 1:
        act = act.squeeze(0) # If batch size is 1, just remove dim
    elif act.ndim == 3:
        print("Assuming shape is already (Channels, H, W).")
        pass # Shape is likely already correct
    else:
        print(f"Activation tensor has unexpected shape: {act.shape}.")
        return

    print(f"Shape after batch selection: {act.shape}") # Should be (Channels, H, W)

    num_channels = act.shape[0]
    num_rows = (num_channels + num_cols - 1) // num_cols

    # Handle cases with very few channels/plots correctly
    if num_rows == 0: num_rows = 1
    if num_channels == 0:
        print("No channels to plot.")
        return
    # If plotting only one row or col, subplots might return a 1D array or single object
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale), squeeze=False)

    axes = axes.flatten() # Flatten to 1D array for easy iteration

    for i in range(num_channels):
        ax = axes[i]
        # Get the individual channel (H, W)
        channel_img = act[i].numpy()

        # Normalize each channel individually for better visualization
        min_val, max_val = channel_img.min(), channel_img.max()
        if max_val > min_val:
            channel_img = (channel_img - min_val) / (max_val - min_val)

        # Now channel_img is (64, 64), which imshow can handle with a colormap
        ax.imshow(channel_img, cmap='viridis')

        ax.set_title(f'Ch {i}')
        ax.axis('off')

    # Turn off axes for any unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"CNN Activations for {layer_name} (Batch {batch_index})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # plt.savefig(f"activations_{layer_name.replace('.', '_')}_b{batch_index}.png")


