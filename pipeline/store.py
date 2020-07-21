"""
Functions for manipulating the general data store.
"""
import os
from typing import Dict, List, Optional

import pandas as pd

from .conversions import CONVERSIONS
from .lib import process_map
from .retrieval import download_to_store, copy_to_store

CLASSES: Dict[str, int] = {
    "Unlabeled": -1,
    "NotGraph": 0,
    "VennDiagram": 1,
    "TreeDiagram": 2,
    "Table": 3,
    "ScatterGraph": 4,
    "RadarPlot": 5,
    "PieChart": 6,
    "ParetoChart": 7,
    "NetworkDiagram": 8,
    "Map": 9,
    "LineGraph": 10,
    "FlowChart": 11,
    "ColumnGraph": 12,
    "BubbleChart": 13,
    "BoxPlot": 14,
    "BarGraph": 15,
    "AreaGraph": 16
}

DEFAULT_CLASS: str = "Unlabeled"


def _init_label_store() -> None:
    """
    Creates a CSV to store information about image data. The included columns,
    listed in logical categories, are:
        --- Data --------------------------------------------------------------
        general information about the file
        File        str     the name of the file
        Class       int     the class value of the image
        --- Flags -------------------------------------------------------------
        Whether the file has been converted or verified to be a certain form.
        Columns in this section have type bool and are generated from the list
        of conversions imported from `conversions.py`.
    :return: None
    """
    df = pd.DataFrame(columns=["File", "Class", *CONVERSIONS.keys()])
    df.to_csv("data/log.csv", index_label="Index")


def init_data_store() -> None:
    """
    If no data store exists, create one.
    :return: None
    """
    try:
        os.mkdir("data")
        os.mkdir("data/images")
        os.mkdir("data/datasets")
        _init_label_store()
    except FileExistsError:
        pass


def import_images(images: List[str], labels: Optional[List[str]] = None,
                  urls: bool = False) -> None:
    """
    Imports images into data store.
    :param images: The list of image path/URLs to import.
    :param labels: The image classes of the images.
    :param urls: Whether or not the images are URLs (otherwise paths).
    :return: None.
    """
    labels = labels if labels else [DEFAULT_CLASS for _ in images]
    filenames = process_map(download_to_store if urls else copy_to_store,
                            images)
    df_old = pd.read_csv("data/log.csv", index_col="Index")
    data = [[f, l, *[False for _ in CONVERSIONS.keys()]] for f, l in
            zip(filenames, labels) if f is not None]
    index_start = max(df_old.index) + 1
    index = list(range(index_start, index_start + len(data)))
    df_new = pd.DataFrame(data, columns=df_old.columns, index=index)
    df = df_old.append(df_new)
    df.to_csv("data/log.csv", index_label="Index")


def _convert_image(image: str, conversions: List[str]) -> str:
    """
    Converts an image in the data store and returns the path to the new image.
    :param image: The path to the image to convert.
    :param conversions: The list of conversions to apply.
    :return: The converted image.
    """
    for c in conversions:
        image = CONVERSIONS[c](image)
    return image


def convert_images(images: List[str], conversions: List[str]) -> None:
    """
    Destructively apply a set of conversions to a set of images in the main
    store.
    :param images: The list of images to work with.
    :param conversions: The list of conversions to apply.
    :return: None.
    """
    df = pd.read_csv("data/log.csv", index_col="Index")

    conversions_left = [
        (r["File"], [c for c in conversions if not r[c]])
        for _, r in df.iterrows() if r["File"] in images
    ]
    new_files = process_map(_convert_image, conversions_left, packed=True)
    for new, (old, c) in zip(new_files, conversions_left):
        df.loc[df["File"] == old, ["File", *c]] = [new, *[True] * len(c)]
    df.to_csv("data/log.csv", index_label="Index")
