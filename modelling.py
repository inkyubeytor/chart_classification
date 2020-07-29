import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from pipeline.dataset import new_dataset, make_data, delete_dataset
from pipeline.store import CLASSES

def train_and_save(classifier: ClassifierMixin, dataset: str,
                   transforms: List[str], bundled: bool,
                   test_proportion: int = 0.1) -> None:
    """
    Trains on the given dataset and saves model.
    :param classifier: The classifier to train.
    :param dataset: The dataset to train on.
    :param transforms: The transforms to apply to the data.
    :param bundled: Whether to bundle chart classes together.
    :param test_proportion: What percentage of the dataset to use for testing.
    :return: None.
    """
    if not make_data(dataset, transforms, bundled):
        raise FileNotFoundError
    images, labels = np.load(f"{dataset}/X.npy"), np.load(f"{dataset}/Y.npy")
    X_train, X_test, Y_train, Y_test = \
        train_test_split(images, labels, test_size=test_proportion)
    classifier.fit(X_train, Y_train)
    pred = classifier.predict(X_test)
    print(classification_report(Y_test, pred))
    print(pd.DataFrame(confusion_matrix(Y_test, pred)))
    joblib.dump(classifier, f"{dataset}/model.joblib")


def load_and_predict(model_dataset: str, test_dataset: str) -> None:
    """
    Loads a model from one dataset and tests it on another. Overwrites the
    data numpy files of the test dataset.
    Note: if a dataset is used as a test dataset, it will not necessarily
    have a model matching the parameters in process.json. For this reason,
    datasets should be separated into model datasets and test datasets, with
    duplication if necessary.
    :param model_dataset: The dataset from which the model comes from.
    :param test_dataset: The dataset to test on.
    :return: None.
    """
    print("Loading model")
    classifier = joblib.load(f"{model_dataset}/model.joblib")
    print("Formatting data")
    p = get_process(model_dataset)
    make_data(test_dataset, p["Transforms"], p["Bundled"])
    images, labels = np.load(f"{test_dataset}/X.npy"), np.load(
        f"{test_dataset}/Y.npy")
    print("Starting classifier")
    pred = classifier.predict(images)
    print(classification_report(labels, pred))
    print(pd.DataFrame(confusion_matrix(labels, pred)))


def export_model(model_dataset: str, dest: str) -> str:
    """
    Exports a model with associated process to a given location.
    :param model_dataset: The path to the model to export.
    :param dest: The destination directory.
    :return: The path to the exported model.
    """
    try:
        os.makedirs(dest)
    except FileExistsError:
        pass

    classifier = joblib.load(f"{model_dataset}/model.joblib")
    process = get_process(model_dataset)
    output = f"{dest}/export_model.joblib"
    joblib.dump((process, classifier), output)
    return output


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
