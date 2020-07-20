"""
A set of tools to process image files. All conversions have the type
`fp: str, dest: Optional[str] = None) -> str`.

To add a conversion, add a function and then add a flag name to the CONVERSIONS
global list.
"""
import os
from typing import Callable, Dict, Optional

from PIL import Image

# Scaling dimensions
HEIGHT = 300
WIDTH = 400


def convert_to_png(fp: str, dest: Optional[str] = None) -> str:
    """
    Converts an image to a PNG.
    :param fp: The path to the image to convert.
    :param dest: A destination path, if a non-destructive operation is desired.
    :return: The path to the converted image.
    """
    new_fp = dest or f"{'.'.join(fp.split('.')[:-1])}.png"

    img = Image.open(fp)
    img.load()  # Force loading of the image into memory
    img.save(new_fp, format="png")
    img.close()

    if fp != new_fp:
        os.remove(fp)

    return new_fp


def make_grayscale(fp: str, dest: Optional[str] = None) -> str:
    """
    Converts an image to single-channel grayscale. In-place if no destination
    path is specified.
    :param fp: The image to convert.
    :param dest: A destination path, if a non-destructive operation is desired.
    :return: The path to the new image.
    """
    img = Image.open(fp)
    img = img.convert("L")
    new_path = dest or fp
    img.save(new_path)
    return new_path


def scale_image(fp: str, dest: Optional[str] = None) -> str:
    """
    Scales an image to HEIGHT x WIDTH, the standard size used by the pipeline
    for modelling.
    :param fp: The image to convert.
    :param dest: A destination path, if a non-destructive operation is desired.
    :return: The path to the new image.
    """
    img = Image.open(fp)
    img = img.resize((WIDTH, HEIGHT))
    new_path = dest or fp
    img.save(new_path)
    return new_path


# List of available conversions
CONVERSIONS: Dict[str, Callable[[str, Optional[str]], str]] = {
    "PNG": convert_to_png,
    "Grayscale": make_grayscale,
    "Size Scaled": scale_image
}
