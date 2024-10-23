"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-10-22
Purpose: Compares performance of various machine learning models such as PCA, 
simulated anneal, and genetic algorithm

Sources:
- ChatGPT (clarification/corrections to population sets, and in particular
  validating business logic of genetic algorithm)
- scikit-learn docs
"""

from typing import Any
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import random


# Utility to print values
def evaluate_model(
    x, y, selected_features, label: str, random_state: int, cv_folds: int = 2
) -> None:
    clf = DecisionTreeClassifier()

    # Perform 2-fold cross-validation and get the predictions
    y_pred = cross_val_predict(clf, x, y, cv=cv_folds)

    # Generate confusion matrix and accuracy for the combined results of both folds
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    print(f"\nPart {label}:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Features used: {selected_features}")


# Part 3: Simulated Annealing for feature selection
def simulated_annealing(
    x_combined, y, iterations: int = 100, c: int = 1, restart_value: int = 10
) -> tuple[Any, np.ndarray[Any, np.dtype[np.signedinteger[Any]]]]:
    current_features = np.random.choice([0, 1], size=x_combined.shape[1])
    best_accuracy = 0
    restart_count = 0
    for iteration in range(iterations):
        if restart_count >= restart_value:
            current_features = np.random.choice([0, 1], size=x_combined.shape[1])
            restart_count = 0

        new_features = current_features.copy()
        perturbation = random.randint(1, 2)
        for _ in range(perturbation):
            idx = random.randint(0, len(current_features) - 1)
            new_features[idx] = 1 - new_features[idx]  # Flip feature on/off

        selected_columns = np.where(new_features == 1)[0]
        if len(selected_columns) == 0:
            continue

        x_selected = x_combined[:, selected_columns]
        accuracy = np.mean(
            cross_val_score(DecisionTreeClassifier(), x_selected, y, cv=2)
        )

        if accuracy > best_accuracy:
            current_features = new_features
            best_accuracy = accuracy
            restart_count = 0
            status = "Improved"
        else:
            pr_accept = np.exp((accuracy - best_accuracy) / c)
            rand_uniform = np.random.uniform()
            if rand_uniform < pr_accept:
                current_features = new_features
                status = "Accepted"
            else:
                status = "Discarded"

        print(
            f"Iteration {iteration}: Features {selected_columns}, Accuracy: {accuracy:.4f}, Status: {status}"
        )

    selected_columns = np.where(current_features == 1)[0]
    return x_combined[:, selected_columns], selected_columns


# Part 4: Genetic Algorithm for feature selection
def genetic_algorithm(x_combined, y, generations=50, population_size=5) -> None:
    # I asked ChatGPT for these population sets. I trust that iris dataset has
    # been done to death so this is probably right

    # fmt: off
    population = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],  # z1, sepal-length, sepal-width, petal-length, petal-width
        [1, 1, 0, 1, 1, 0, 0, 0],  # z1, z2, sepal-width, petal-length, petal-width
        [1, 1, 1, 0, 1, 0, 0, 0],  # z1, z2, z3, sepal-width, petal-length
        [1, 1, 1, 1, 0, 0, 0, 1],  # z1, z2, z3, z4, sepal-width
        [1, 1, 1, 1, 1, 1, 0, 0],  # z1, z2, z3, z4, sepal-length
    ])
    # fmt: on

    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            selected_columns = np.where(individual == 1)[0]
            x_selected = x_combined[:, selected_columns]
            accuracy = np.mean(
                cross_val_score(DecisionTreeClassifier(), x_selected, y, cv=2)
            )
            fitness_scores.append((accuracy, individual))

        fitness_scores = sorted(fitness_scores, key=lambda x: x[0], reverse=True)

        print(f"\nGeneration {generation}:")
        for i in range(population_size):
            selected_columns = np.where(fitness_scores[i][1] == 1)[0]
            print(f"Features: {selected_columns}, Accuracy: {fitness_scores[i][0]:.4f}")

        # Selection (top 2) + Crossover
        parent1, parent2 = fitness_scores[0][1], fitness_scores[1][1]
        crossover_point = random.randint(1, len(parent1) - 2)
        child1 = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.hstack((parent2[:crossover_point], parent1[crossover_point:]))

        # Mutation
        mutation_prob = 0.1
        if random.random() < mutation_prob:
            idx = random.randint(0, len(child1) - 1)
            child1[idx] = 1 - child1[idx]
        if random.random() < mutation_prob:
            idx = random.randint(0, len(child2) - 1)
            child2[idx] = 1 - child2[idx]

        population = [fitness_scores[i][1] for i in range(3)] + [child1, child2]


def run() -> None:
    seed = 658  # seed everything for reproducibility
    random.seed(seed)

    iris = load_iris()
    x, y = iris.data, iris.target
    features = iris.feature_names

    ########
    # Part 1
    ########
    evaluate_model(x, y, features, 1, seed)

    ########
    # Part 2
    ########
    pca = PCA(n_components=4)
    x_pca = pca.fit_transform(x)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    pov = np.cumsum(pca.explained_variance_ratio_)

    print("\nPart 2:")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors: {eigenvectors}")
    print(f"PoV: {pov}")

    # Subset where PoV > 0.90
    selected_pca_indices = np.where(pov > 0.9)[0]
    x_pca_selected = x_pca[:, : selected_pca_indices[0] + 1]
    evaluate_model(
        x_pca_selected,
        y,
        [f"z{i+1}" for i in range(selected_pca_indices[0] + 1)],
        2,
        seed,
    )

    ########
    # Part 3
    ########
    # Combine original 4 features + 4 PCA features
    x_combined = np.hstack((x, x_pca))
    x_sim_anneal, sim_features = simulated_annealing(x_combined, y)
    evaluate_model(
        x_sim_anneal,
        y,
        [features[i] if i < 4 else f"z{i-3}" for i in sim_features],
        3,
        seed,
    )

    # Part 4
    genetic_algorithm(x_combined, y)


if __name__ == "__main__":
    run()
