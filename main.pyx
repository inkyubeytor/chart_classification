from typing import List

import joblib
import numpy as np

from pipeline.dataset import new_dataset, make_data, delete_dataset
from pipeline.store import CLASSES


def end_to_end_prediction(exported_model: str, image_paths: List[str]) \
        -> List[str]:
    """
    Loads the exported model and process, converts the given images to a
    dataset with the same process, classifies the images, deletes the dataset,
    and returns the result.
    :param exported_model: A model exported with `export_model`.
    :param image_paths: Paths to image
    :return:
    """
    process, classifier = joblib.load(exported_model)
    dataset = new_dataset(image_paths, process["Conversions"], from_store=False)
    if not make_data(dataset, process["Transforms"], process["Bundled"]):
        raise FileNotFoundError
    images = np.load(f"{dataset}/X.npy")
    pred = classifier.predict(images)
    delete_dataset(dataset)
    if process["Bundled"]:
        return ["Graph" if p else "NotGraph" for p in pred]
    else:
        return [list(CLASSES.keys())[list(CLASSES.values()).index(p)]
                for p in pred]