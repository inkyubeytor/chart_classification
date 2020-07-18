"""
Functions for constructing datasets.
"""
from typing import Callable, List

import numpy as np
from PIL import Image

from .lib import process_map


def load_image_array(fp: str) -> np.ndarray:
    """
    Converts an image on disk to a numpy array.
    :param fp: The image to convert.
    :return: The image data as a numpy array.
    """
    img = Image.open(fp)
    arr = np.array(img)
    img.close()
    return arr


def make_imageset(fps: List[str], transforms: List[Callable],
                  dataset: str) -> bool:
    """
    Loads the images from a set of URLS, applies a series of transforms, and
    saves the result to the dataset.
    :param fps: A list of image file paths.
    :param transforms: A list of transform functions to apply when loading.
    :param dataset: The path to the dataset.
    :return: Whether the operation was successful.
    """
    try:
        images = process_map(load_image_array, fps)
    except FileNotFoundError:
        return False
    for f in transforms:
        images = process_map(f, images)
    np.save(f"{dataset}/X.npy", np.array(images))
    return True
