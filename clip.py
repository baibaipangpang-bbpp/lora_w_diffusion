import torch

def calculate_clip_score(image, prompt, model, processor, device):
    """
    Calculates the CLIP score between a given image and a text prompt.

    Args:
        image (PIL.Image.Image or torch.Tensor): The input image. Can be a
            PIL Image object or a PyTorch tensor representing the image.
        prompt (str): The text prompt to compare against the image.
        model (torch.nn.Module): The pre-trained CLIP model.
        processor (transformers.CLIPProcessor): The CLIP processor associated
            with the model for preparing inputs.
        device (torch.device): The device (CPU or GPU) on which the model
            is loaded.

    Returns:
        float: The CLIP score, representing the cosine similarity between the
               image and text embeddings. Higher scores indicate a stronger
               semantic alignment between the image and the prompt.
    """
    inputs = processor(
        text=[prompt], images=image, return_tensors="pt", padding=True
    ).to(device)
  
    with torch.no_grad():
        outputs = model(**inputs)
  
    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds
  
    image_embeddings_norm = image_embeddings / image_embeddings.norm(
        dim=-1, keepdim=True
    )
    text_embeddings_norm = text_embeddings / text_embeddings.norm(
        dim=-1, keepdim=True
    )
  
    clip_score = torch.matmul(
        image_embeddings_norm, text_embeddings_norm.T
    ).item()
  
    return clip_score