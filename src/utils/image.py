import numpy as np
import torch
from PIL import Image


def save_image(image: torch.Tensor, path: str, normalize: bool = True):
    """Save an image to a file."""
    # print(image.max())
    # print(image.min())
    image = image.detach().cpu().numpy()
    if normalize:
        image = (image + 0.5) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = image.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    pil_image.save(path)


def save_gif_video(images: np.ndarray, path: str, fps: int = 30):
    """Save a list of images as a gif video."""
    images = (images + 0.5) * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(
        path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000 // fps,
        loop=0,
    )
