"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-10-31
Purpose: Compares imbalanced data set accuracies with various models and 
methodologies such as oversampling/undersampling

Sources:
- ChatGPT (clarification/corrections specifically with MLPClassifier warnings 
  on nonconvergence)
  - as an aside, i feel like theres gotta a be a better way rather than
    increasing max_iter to sufficiently large number? but it doesn't take up any
    more processing time on my computer? so its all a wash? seems unintelligent.
    like isn't this overfitting? oh whale 🐋
  - learning_rate="adaptive" counteracts this behavior by only doing iters that 
    decrease training loss? so maybe this is a sufficient fix? w/e...
- scikit-learn docs
- imblearn docs (https://imbalanced-learn.org)

Other:
i need to use something like blue, black doesn't respect that this is column 80|
it uses line 88? thats arbitrary. might as well set it to 9 quintillion (2^63-1)
"""

from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks

SEED = 658  # really this should be a cmdline argument or something... whatever

data = pd.read_csv("imbalanced iris.csv")
x = data.drop("class", axis=1)
y = data["class"]

# classifiers
# increased max_iter and changed learning rate to adaptive to ensure convergence
clf = MLPClassifier(max_iter=1000, random_state=SEED, learning_rate="adaptive")
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
# impure programming is my #1 friend


# Part 1: Imbalanced Data Set
def part1() -> None:
    print("Part 1: Imbalanced Data Set")
    for train_idx, test_idx in skf.split(x, y):
        X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Confusion Matrix and Accuracy
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        custom_class_balanced_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        custom_balanced_accuracy = custom_class_balanced_acc.mean()

        print("Confusion Matrix:\n", conf_matrix)
        print("Accuracy:", accuracy)
        # i didn't like the way this formatted... too much work, we know it goes
        # in this order... i think?
        # fmt: off
        # print("Custom Class Balanced Accuracy:\n\tsetosa\tversicolor\tvirginica\n\t", "\t".join([str("{0:.2f}".format(cl)) for cl in custom_class_balanced_acc]))
        # fmt: on
        print("Custom Class Balanced Accuracy:", custom_class_balanced_acc)
        print("Custom Balanced Accuracy:", custom_balanced_accuracy)
        print("Balanced Accuracy (sklearn):", balanced_accuracy_score(y_test, y_pred))
        print()


# Part 2: Oversampling
def part2() -> None:
    print("Part 2: Oversampling")
    oversamplers: dict[str, Any] = {
        "Random Oversampling": RandomOverSampler(random_state=SEED),
        "SMOTE": SMOTE(random_state=SEED),
        # almost missed minority here
        "ADASYN": ADASYN(sampling_strategy="minority", random_state=SEED),
    }

    for method, sampler in oversamplers.items():
        X_res, y_res = sampler.fit_resample(x, y)
        y_pred = cross_val_predict(clf, X_res, y_res, cv=skf)

        conf_matrix = confusion_matrix(y_res, y_pred)
        accuracy = accuracy_score(y_res, y_pred)

        print(f"{method} Confusion Matrix:\n", conf_matrix)
        print(f"{method} Accuracy:", accuracy)
        print()


# Part 3: Undersampling
def part3() -> None:
    print("Part 3: Undersampling")
    undersamplers: dict[str, Any] = {
        "Random Undersampling": RandomUnderSampler(random_state=SEED),
        "Cluster Centroids": ClusterCentroids(random_state=SEED),
        "Tomek Links": TomekLinks(),
    }

    # LOL? do we know how to dry code? not in this house
    for method, sampler in undersamplers.items():
        X_res, y_res = sampler.fit_resample(x, y)
        y_pred = cross_val_predict(clf, X_res, y_res, cv=skf)

        conf_matrix = confusion_matrix(y_res, y_pred)
        accuracy = accuracy_score(y_res, y_pred)

        print(f"{method} Confusion Matrix:\n", conf_matrix)
        print(f"{method} Accuracy:", accuracy)
        print()


def run() -> None:
    part1()
    part2()
    part3()


if __name__ == "__main__":
    # raise NotImplementedError()
    run()
