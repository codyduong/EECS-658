"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-09-02
Purpose: Naive Baysian Classifier

Resources utilized:
* https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
* https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
* https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
* https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold.split
* https://scikit-learn.org/stable/api/sklearn.metrics.html
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

"""

from os import path
from typing import Any, List, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # type: ignore
from sklearn.utils import Bunch


# i think i typed this right...
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
class IrisBunch(Bunch):
    # lol i give up i don't know what these data structs are
    data: Union[np.ndarray[Any, Any], pd.DataFrame]  # type: ignore
    target: Union[np.ndarray[Any, Any], pd.Series]  # type: ignore
    feature_names: List[str]
    target_names: List[str]
    frame: pd.DataFrame
    DESCR: str
    filename: str


def run() -> None:
    # iris = cast(IrisBunch, datasets.load_iris())
    # x = iris.data
    # y = iris.target
    # target_names = iris.target_names
    # oh huh I wrote this before realizing there is a .csv provided...

    # I used ChatGPT to find how to read this into x and y
    df = pd.read_csv(path.join(path.dirname(path.abspath(__file__)), "iris.csv"), header=None)  # type: ignore
    x = df.iloc[:, :-1].values  # type: ignore
    y = df.iloc[:, -1].values
    target_names: Any = df.iloc[:, -1].unique()  # type: ignore

    nb = GaussianNB()

    kf = KFold(
        n_splits=2, shuffle=True, random_state=568
    )  # 2 splits, ensure we always seed the same
    predicted = np.array([])
    actual = np.array([])

    # Divide and train each fold
    # Cross-validation for K-folds here: https://machinelearningmastery.com/k-fold-cross-validation/
    for train_index, test_index in kf.split(x):  # type: ignore
        x_train, x_test = x[train_index], x[test_index]  # type: ignore
        y_train, y_test = y[train_index], y[test_index]

        nb.fit(x_train, y_train)  # type: ignore

        y_pred = nb.predict(x_test)  # type: ignore

        # Concatenate the fold results
        predicted = np.concatenate([predicted, y_pred])  # type: ignore
        actual = np.concatenate([actual, y_test])

    print(
        f"""Overall Accuracy: {accuracy_score(actual, predicted):.2f}
Confusion Matrix:\n{confusion_matrix(actual, predicted)}
Classification Report:\n{classification_report(actual, predicted, target_names=target_names)}
"""
    )


if __name__ == "__main__":
    run()
