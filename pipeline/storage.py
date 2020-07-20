"""
Functions for initializing and deleting components of the storage system.
"""
import os
import json
import itertools
import shutil
from typing import Dict

import pandas as pd

from .conversions import CONVERSIONS

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


def new_dataset() -> str:
    """
    Initialize a new dataset.
    :return: The path to the dataset folder.
    """
    datasets = os.listdir("data/datasets")
    i = next(i for i in itertools.count() if f"dataset-{i}" not in datasets)
    path = f"data/datasets/dataset-{i}"
    os.mkdir(path)
    os.mkdir(f"{path}/images")
    df = pd.DataFrame(columns=["File", "Class"])
    df.to_csv(f"{path}/log.csv", index_label="Index")
    with open(f"{path}/process.json", "w+") as f:
        json.dump({"Conversions": [], "Transforms": []}, f)
    return path


def delete_dataset(dataset: str) -> None:
    """
    Delete a dataset.
    :param dataset: The path to the dataset to delete.
    :return: None
    """
    shutil.rmtree(dataset)
