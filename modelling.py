from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
import numpy as np
import pandas as pd
from pipeline.dataset import make_data, get_process
from typing import List
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def train_and_save(classifier: ClassifierMixin, dataset: str,
                   transforms: List[str], bundled: bool) -> None:
    """
    Trains on the given dataset and saves model.
    :param classifier: The classifier to train.
    :param dataset: The dataset to train on.
    :param transforms: The transforms to apply to the data.
    :param bundled: Whether to bundle chart classes together.
    :return: None.
    """
    make_data(dataset, transforms, bundled)
    images, labels = np.load(f"{dataset}/X.npy"), np.load(f"{dataset}/Y.npy")
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels)
    classifier.fit(X_train, Y_train)
    pred = classifier.predict(X_test)
    print(classification_report(Y_test, pred))
    print(pd.DataFrame(confusion_matrix(Y_test, pred)))
    joblib.dump(classifier, f"{dataset}/model.joblib")


def load_and_predict(model_dataset: str, test_dataset: str) -> None:
    """
    Loads a model from one dataset and tests it on another. Overwrites the
    data numpy files of the test dataset.
    :param model_dataset: The dataset from which the model comes from.
    :param test_dataset: The dataset to test on.
    :return: None.
    """
    classifier = joblib.load(f"{model_dataset}/model.joblib")
    p = get_process(model_dataset)
    make_data(test_dataset, p["Transforms"], p["Bundled"])
    images, labels = np.load(f"{test_dataset}/X.npy"), np.load(f"{test_dataset}/Y.npy")
    pred = classifier.predict(images)
    print(classification_report(labels, pred))
    print(pd.DataFrame(confusion_matrix(labels, pred)))
