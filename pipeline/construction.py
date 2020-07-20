"""
Functions for constructing datasets.
"""
from typing import List
import json

import numpy as np
import pandas as pd
from PIL import Image

from .lib import process_map
from .transforms import TRANSFORMS
from .storage import CLASSES


def _load_image_array(fp: str) -> np.ndarray:
    """
    Converts an image on disk to a numpy array.
    :param fp: The image to convert.
    :return: The image data as a numpy array.
    """
    img = Image.open(fp)
    arr = np.array(img)
    img.close()
    return arr


def _make_imageset(dataset: str, transforms: List[str]) -> bool:
    """
    Loads the images from dataset image store, applies a series of transforms,
    and saves the result to the dataset.
    :param transforms: A list of transform functions to apply when loading.
    :param dataset: The path to the dataset.
    :return: Whether the operation was successful.
    """
    try:
        df = pd.read_csv(f"{dataset}/log.csv")
        fps = list(f"{dataset}/images/{f}" for f in df["File"])
        images = process_map(_load_image_array, fps)
    except FileNotFoundError:
        return False
    for f in transforms:
        images = process_map(TRANSFORMS[f], images)
    with open(f"{dataset}/process.json", "r+") as f:
        data = json.load(f)
        data["Transforms"] = transforms
        json.dump(data, f)
    np.save(f"{dataset}/X.npy", np.array(images))
    return True


def _make_labelset(dataset: str, bundled: bool = True) -> bool:
    """
    Turns the labels of a dataset into training data labels, applying bundling
    of chart classes if desired.
    :param dataset: The dataset to create label data for.
    :param bundled: Whether the chart classes should be bundled.
    :return: Whether the operation was successful.
    """
    df = pd.read_csv(f"{dataset}/log.csv")
    classes = [int(bool(CLASSES[c])) if bundled else CLASSES[c] for c in
               df["Classes"]]
    np.save(f"{dataset}/Y.npy", np.array(classes))
    return True


def make_data(dataset: str, transforms: List[str],
              bundled: bool = True) -> bool:
    """
    Construct X.npy and Y.npy dataset files.
    :param dataset: The dataset to convert.
    :param transforms: The list of transforms to apply to the images.
    :param bundled: Whether the label classes should be bundled.
    :return:
    """
    return _make_imageset(dataset, transforms) and \
        _make_labelset(dataset, bundled)
