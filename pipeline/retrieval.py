"""
Functions for retrieving images.
"""
import imghdr
import itertools
import os
import random
import shutil
import time
from typing import List, Optional

import pandas as pd
import requests

from .conversions import CONVERSIONS
from .lib import process_map
from .storage import DEFAULT_CLASS

IMAGE_FORMATS = ["jpg", "jpeg", "png", "gif", "tiff", "tif", "bmp"]


def _get_filetype_from_name(name: str) -> Optional[str]:
    """
    Checks whether a file name is an image file name.
    :param name: The filename to check (not a path).
    :return: The file extension if it is a valid image extension..
    """
    extension = name.split('.')[-1]
    if extension in IMAGE_FORMATS:
        return extension
    else:
        return None


def _check_filetype(fp: str, extension: str) -> bool:
    """
    Checks whether a file is an image file encoding.
    :param fp: The file to check.
    :return: True if the file is an image
    """
    extension = "jpeg" if extension == "jpg" else extension
    return imghdr.what(fp) == extension


def _generate_file_name() -> str:
    """
    Generates a "unique" filename/ID for an image.
    :return: The image name.
    """
    return f"{time.time_ns()}{random.randint(100000, 999999)}"


def _copy_to_store(fp: str) -> Optional[str]:
    """
    Copies an image to data folder.
    :param fp: The image to copy.
    :return: A path to the new image.
    """
    name = os.path.basename(fp)
    extension = _get_filetype_from_name(name)
    if extension and _check_filetype(fp, extension):
        new_path = f"data/images/{_generate_file_name()}.{extension}"
        shutil.copyfile(fp, new_path)
        return new_path
    else:
        return None


def _download_to_store(url: str) -> Optional[str]:
    """
    Downloads an image from a URL to data folder.
    :param url: The URL of the image to download.
    :return: A path to the new image.
    """
    name = url.split('/')[-1].lower()
    extension = _get_filetype_from_name(name)
    if extension:
        response = requests.get(url)
        new_path = f"data/images/{_generate_file_name()}.{extension}"
        with open(new_path, "wb") as f:
            f.write(response.content)
        if _check_filetype(new_path, extension):
            return new_path
        else:
            os.remove(new_path)
            return None


def import_images(images: List[str], labels: Optional[List[str]] = None,
                  urls: bool = False) -> None:
    """
    Imports images into a dataset.
    :param images: The list of image path/URLs to import.
    :param labels: The image classes of the images.
    :param urls: Whether or not the images are URLs (otherwise paths).
    :return: None.
    """
    labels = labels if labels else [DEFAULT_CLASS for _ in images]
    filenames = process_map(_download_to_store if urls else _copy_to_store,
                            images)
    df_old = pd.read_csv("data/log.csv", index_col="Index")
    data = [[f, l, *[False for _ in CONVERSIONS.keys()]] for f, l in
            zip(filenames, labels) if f is not None]
    index = list(range(
        max(df_old.index) + 1, max(df_old.index) + 1 + len(data)
    ))
    df_new = pd.DataFrame(data, columns=df_old.columns, index=index)
    df = df_old.append(df_new)
    df.to_csv("data/log.csv", index_label="Index")
