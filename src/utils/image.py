import numpy as np
from PIL import Image


def save_image(image: np.ndarray, path: str):
    """Save an image to a file."""
    image = image * 255
    image = image.astype(np.uint8)
    image = image.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    pil_image.save(path)


def save_gif_video(images: np.ndarray, path: str, fps: int = 30):
    """Save a list of images as a gif video."""
    images = images * 255
    images = images.astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(
        path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000 // fps,
        loop=0,
    )
