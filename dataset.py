import os

# library imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, root_dir="data"):
        self.root_dir = root_dir
        self.samples = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith((".jpg", ".png")):
                basename = os.path.splitext(fname)[0]
                txt_path = os.path.join(root_dir, basename + ".txt")
                if os.path.exists(txt_path):
                    self.samples.append((os.path.join(root_dir, fname), txt_path))

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        with open(txt_path, "r") as f:
            caption = f.read().strip()
        return image, caption