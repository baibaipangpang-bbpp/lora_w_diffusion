from PIL import Image
import os

input_dir = "~/Desktop/tmp2"
output_dir = "~/Desktop/output"
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(input_dir, filename)
        img = Image.open(path)
        img = img.resize((1024, 1024))  # or use img.thumbnail() for aspect ratio
        img.save(path)
