import os

import pandas as pd

from .image import CONVERSIONS


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
        of conversions imported from `image.py`.
    :return: None
    """
    df = pd.DataFrame(columns=["File", "Class", *CONVERSIONS])
    df.to_csv("data/images.csv")
