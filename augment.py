import kornia.augmentation as K
import torch.nn as nn

class CalligraphyAugmentations(nn.Module):
    def __init__(self):
        super().__init__()
        self.augment = nn.Sequential(
            K.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),             # exposure variation
            K.RandomGrayscale(p=0.2),                                       # ensure grayscale
            K.RandomRotation(degrees=3.0, p=0.5),
            # K.RandomResizedCrop((512, 512), scale=(0.95, 1.05), ratio=(1.0, 1.0), p=0.5),
            K.RandomGaussianBlur((3, 3), sigma=(0.1, 1.0), p=0.5),
            # K.RandomElasticTransform(kernel_size=(63, 63), sigma=(4.0, 6.0), alpha=(1.0, 2.0), p=0.3),
            # K.RandomErasing(scale=(0.01, 0.03), ratio=(0.3, 3.3), same_on_batch=False, p=0.2),
        )

    def forward(self, x):
        return self.augment(x)

device = "cuda"
augment = CalligraphyAugmentations().to(device)
augment.eval()