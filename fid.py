import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from matplotlib import pyplot as plt

def load_images_from_folder(root_dir, transform, device, show_example = False):
    """
    Loads images from a specified folder, applies a transformation,
    and prepares them as tensors for further processing.

    Args:
        root_dir (str): The path to the directory containing the images.
        transform (callable): A torchvision.transforms object to apply to each image.
                              This transformation should handle resizing and any
                              necessary preprocessing steps (e.g., normalization).
        device (torch.device): The torch device to move the output tensors to (e.g., 'cuda' or 'cpu').
        show_example (bool, optional): If True, displays the first loaded and
                                       transformed image using matplotlib. Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
            - list: A list of PIL Image objects after the initial transformation
                    (e.g., resizing). This is primarily for visualization purposes.
            - torch.Tensor: A tensor containing all the loaded and transformed images,
                           stacked along the first dimension (batch dimension).
                           The images are also moved to the specified 'device'.
                           Returns (None, None) if no valid image files are found.
    """
    images = [] # store the original images for visualization
    image_tensors = [] # output image tensor for computing fid score
    tensor = transforms.ToTensor()
    for fname in os.listdir(root_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root_dir, fname)
            image = Image.open(path).convert('RGB') # Ensure image is RGB

            # Resize and transform to grayscale
            image = transform(image)
            images.append(image)

            # Transform to tensor for score calculation
            image_tensor = tensor(image).to(device)
            image_tensors.append(image_tensor)

            if show_example:
              print (f"Processed image {fname}:")
              plt.imshow(image)
              plt.show()
              show_example = False
  
    if not image_tensors:
        return None, None
    return images, torch.stack(image_tensors).to(device)

def calculate_fid_score(fake_images, real_images, device):
    """
    Calculates the Frechet Inception Distance (FID) between two sets of images.

    FID is a metric used to evaluate the quality of generated images by comparing
    the distribution of their features with the distribution of features from a
    set of real images. Lower FID scores indicate higher similarity and better
    quality of the generated images.

    Args:
        fake_images (torch.Tensor): A tensor of generated images with shape (N, C, H, W),
                                  where N is the number of images, C is the number of
                                  channels (e.g., 3 for RGB), and H and W are the
                                  height and width of the images. The pixel values
                                  should be in the range [0, 255] and have a dtype of
                                  torch.uint8 or be normalized to [0.0, 1.0].
        real_images (torch.Tensor): A tensor of real images with the same shape and
                                  pixel value range as `fake_images`.
        device (torch.device): The torch device to perform the calculations on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A scalar tensor containing the calculated FID score.
    """
    # normalize=True otherwise it gives uint64
    fid = FrechetInceptionDistance(normalize=True).to(device)  # Initialize and move to device
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()

def add_fake_data_and_calculate_fid_score(real_images_fid, fake_images, device):
    """
    Updates an existing Frechet Inception Distance (FID) calculator with a batch
    of fake (generated) images and then computes the FID score.

    This function assumes that the FID calculator has already been initialized
    and updated with a batch of real images. It then adds the statistics
    of the provided fake images to the calculator and computes the final FID score
    based on the accumulated real and fake image statistics.
    """
    real_images_fid.update(fake_images, real=False)
    return real_images_fid.compute()

def calculate_fid_score_between_folders(fake_image_path, real_image_path, transform, device, show_example = False):
    real_images, real_torch = load_images_from_folder(real_image_path, transform, device, show_example)
    fake_images, fake_torch = load_images_from_folder(fake_image_path, transform, device, show_example)
    fid_score = calculate_fid_score(real_torch, fake_torch, device)
    return fid_score
