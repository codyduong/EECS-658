"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-09-17
Purpose: Compare ML Models
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import confusion_matrix, accuracy_score

# thanks for providing all the scikit-learn names equivalent
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html

models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (degree 2)": make_pipeline(
        PolynomialFeatures(2), LinearRegression()
    ),
    "Polynomial Regression (degree 3)": make_pipeline(
        PolynomialFeatures(3), LinearRegression()
    ),
    "Naive Bayes": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
}


def compare() -> None:
    iris = load_iris()
    x = iris.data
    y = iris.target

    kf = KFold(
        n_splits=2, shuffle=True, random_state=568
    )  # 2 splits, ensure we always seed the same

    results = []

    for name, model in models.items():
        all_predictions = []
        all_true_labels = []

        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if "Polynomial" in name:
                # this was less easy, but here: https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples
                degree = 2 if "degree 2" in name else 3
                poly = PolynomialFeatures(degree=degree)
                x_train_poly = poly.fit_transform(x_train)
                x_test_poly = poly.transform(x_test)
                model.fit(x_train_poly, y_train)
                predictions = model.predict(x_test_poly)
            else:
                # huh this is easy
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)

            # we need to clip the nondiscrete models into the three classes
            # otherwise we have a really large nondiscrete (continuous) confusion matrix
            predictions = (
                np.clip(predictions.round().astype(int), 0, 2)
                if "Regression" in name
                else predictions
            )

            all_predictions.extend(predictions)
            all_true_labels.extend(y_test)

        # huh i didn't have to do all that work in Assingment1
        cm = confusion_matrix(all_true_labels, all_predictions)
        acc = accuracy_score(all_true_labels, all_predictions)

        results.append((name, cm, acc))

        if sum(sum(cm)) != 150:
            raise Exception("oops")

    # sort them so its easier to compare
    results = sorted(results, key=lambda x: x[2], reverse=True)
    for name, cm, acc in results:
        print(f"--- {name} ---")
        print("Confusion Matrix:\n", cm)
        print("Accuracy:", acc)
        print()


if __name__ == "__main__":
    compare()
