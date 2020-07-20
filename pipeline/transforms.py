"""
A set of tools to process image arrays. All transforms have the type
`(arr: np.ndarray) -> np.ndarray`.

To add a transform, add a function and then add a flag name to the TRANSFORMS
global dictionary.
"""

from typing import Callable, Dict

import numpy as np


def scale_pixels(arr: np.ndarray) -> np.ndarray:
    """
    Scales the pixel values of an image array to the range [0, 1]. Always
    returns an out-of-place copy.
    :param arr: An array of pixel values representing an image.
    :return: An array representing the same image with pixel values in [0, 1].
    """
    arr = arr.astype("float32")
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


# List of available transforms
TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Scale Pixels": scale_pixels,
    "Flatten": flatten
}
