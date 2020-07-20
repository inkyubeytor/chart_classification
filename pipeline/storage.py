"""
Functions for initializing and deleting components of the storage system.
"""
import os
import json
import itertools
import shutil
from typing import Dict, List

import pandas as pd

from .lib import process_map
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


def new_dataset(filenames: List[str], conversions: List[str]) -> str:
    """
    Create a new dataset from a set of files and conversions.
    :return: The path to the dataset folder.
    """
    # Create new dataset
    datasets = os.listdir("data/datasets")
    i = next(i for i in itertools.count() if f"dataset-{i}" not in datasets)
    dataset = f"data/datasets/dataset-{i}"
    os.mkdir(dataset)
    os.mkdir(f"{dataset}/images")
    with open(f"{dataset}/process.json", "w+") as f:
        json.dump({"Conversions": conversions, "Transforms": []}, f)

    # Add images
    df_store = pd.read_csv("data/log.csv", index_col="Index")
    df = df_store[[f in filenames for f in df_store["File"]]]
    conversions_left = [
        (r["File"], [c for c in conversions if not r[c]])
        for _, r in df.iterrows()
    ]

    def _copy_and_apply(file: str, conversions_to_apply: List[str]) -> None:
        """
        Copies a file to a dataset and applies conversions.
        :param file: The file to copy and process.
        :param conversions_to_apply: The conversions to apply after copying.
        :return: None.
        """
        img = f"{dataset}/images/{file}"
        shutil.copyfile(f"data/images/{file}", img)
        for c in conversions_to_apply:
            img = CONVERSIONS[c](img)

    process_map(_copy_and_apply, conversions_left, packed=True)
    df = df["File", "Class"]
    df.to_csv(f"{dataset}/log.csv", index_label="Index")
    return dataset


def delete_dataset(dataset: str) -> None:
    """
    Delete a dataset.
    :param dataset: The path to the dataset to delete.
    :return: None
    """
    shutil.rmtree(dataset)
