"""
Functions for creating and updating reccords.
"""
import os
import itertools

import pandas as pd

from .conversions import CONVERSIONS
from .transforms import TRANSFORMS


def init_data_store() -> None:
    """
    If no data store exists, create one.
    :return: None
    """
    try:
        os.mkdir("data")
        os.mkdir("data/images")
        os.mkdir("data/datasets")
    except FileExistsError:
        pass


def init_label_store() -> None:
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
    df = pd.DataFrame(columns=["File", "Class", *CONVERSIONS])
    df.to_csv("data/log.csv")


def new_dataset() -> str:
    """
    Initialize a new dataset.
    :return: The name of the dataset folder.
    """
    datasets = os.listdir("data/datasets")
    i = next(i for i in itertools.count() if f"dataset-{i}" not in datasets)
    path = f"data/datasets/dataset-{i}"
    os.mkdir(path)
    os.mkdir(f"{path}/images")
    df = pd.DataFrame(
        columns=["File", "Class", *(CONVERSIONS.keys()), *(TRANSFORMS.keys())])
    df.to_csv(f"{path}/log.csv")
    return f"dataset-{i}"
