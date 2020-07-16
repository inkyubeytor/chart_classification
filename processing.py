import os
import shutil
from typing import Optional
import numpy as np
from PIL import Image

# Scaling dimensions
HEIGHT = 300
WIDTH = 400


def _create_tmp() -> None:
    """
    If no temporary folder exists, create one.
    :return: None.
    """
    try:
        os.mkdir("tmp")
    except FileExistsError:
        pass


def copy_to_tmp(fp: str) -> str:
    """
    Copies an image to temporary folder.
    :param fp: The image to copy.
    :return: A path to the new image.
    """
    new_path = f"tmp/{os.path.basename(fp)}"
    try:
        shutil.copyfile(fp, new_path)
    except FileNotFoundError:
        _create_tmp()
        shutil.copyfile(fp, new_path)
    return new_path


def convert_to_png(fp: str) -> str:
    """
    Destructively converts an image to a PNG.
    :param fp: The path to the image to convert.
    :return: The path to the converted image.
    """
    new_fp = f"{'.'.join(fp.split('.')[:-1])}.png"

    img = Image.open(fp)
    img.load()  # Force loading of the image into memory
    img.save(new_fp, format="png")
    img.close()

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


def make_array(fp: str) -> np.ndarray:
    """
    Converts an image on disk to a numpy array.
    :param fp: The image to convert.
    :return: The image data as a numpy array.
    """
    img = Image.open(fp)
    arr = np.array(img)
    img.close()
    return arr


def scale_pixels(arr: np.ndarray) -> np.ndarray:
    """
    Scales the pixel values of an image array to the range [0, 1]. Always
    returns an out-of-place copy.
    :param arr: An array of pixel values representing an image.
    :return: An array representing the same image with pixel values in [0, 1].
    """
    if arr.max(initial=0.0) > 1.0:
        return arr / 255.0
    else:
        return arr / 1.0


def flatten(arr: np.ndarray) -> np.ndarray:
    """
    Flattens the given image array.
    :param arr: An array representing an image.
    :return: A copy of the array collapsed into one dimension.
    """
    return arr.flatten()
