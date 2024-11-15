"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-11-14
Purpose: Part 1: k-Means Clustering

- ChatGPT helped setup/make matplotlib graphs, esp. in part 3 with U-Matrix/quantization error against grid size
- ^ As an aside I now just saw `PlottingCode.py` was provided to us... w/e. Looks similiar enough.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


# Helper function to match labels
def match_labels(true_labels, cluster_labels):
    matched_labels = np.zeros_like(cluster_labels)
    for i in range(np.max(cluster_labels) + 1):
        mask = cluster_labels == i
        if np.any(mask):
            matched_labels[mask] = mode(true_labels[mask])[0]
    return matched_labels


def part1(seed, x, y) -> None:
    # Part 1: k-Means Clustering
    print("\nPart 1:")

    # Run k-means for k = 1 to 20 and calculate reconstruction error
    reconstruction_errors = []
    k_values = range(1, 21)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(x)
        reconstruction_errors.append(kmeans.inertia_)

    # Plot reconstruction error vs. k
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, reconstruction_errors, marker="o")
    plt.title("k-Means Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Reconstruction Error")
    plt.show(block=False)

    elbow_k = int(input("Enter the elbow_k observed from the plot: "))

    # Use predict() method for k=elbow_k
    kmeans_elbow = KMeans(n_clusters=elbow_k, random_state=seed)
    labels_elbow = kmeans_elbow.fit_predict(x)

    # If elbow_k == 3, calculate confusion matrix and accuracy
    if elbow_k == 3:
        matched_labels_elbow = match_labels(y, labels_elbow)
        cm_elbow = confusion_matrix(y, matched_labels_elbow)
        acc_elbow = accuracy_score(y, matched_labels_elbow)
        print("Confusion Matrix for k=elbow_k:")
        print(cm_elbow)
        print("Accuracy for k=elbow_k:", acc_elbow)
    else:
        print(
            "Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters."
        )
        cm_elbow = confusion_matrix(y, labels_elbow)
        print("Confusion Matrix for k=elbow_k:")
        print(cm_elbow)

    # Use predict() method for k=3
    kmeans_3 = KMeans(n_clusters=3, random_state=seed)
    labels_3 = kmeans_3.fit_predict(x)
    matched_labels_3 = match_labels(y, labels_3)
    cm_3 = confusion_matrix(y, matched_labels_3)
    acc_3 = accuracy_score(y, matched_labels_3)
    print("Confusion Matrix for k=3:")
    print(cm_3)
    print("Accuracy for k=3:", acc_3)


def part2(seed: int, x, y) -> None:
    # Part 2: Gaussian Mixture Models
    print("\nPart 2:")

    # Run GMM for k = 1 to 20 and calculate AIC and BIC
    aic_scores = []
    bic_scores = []
    k_values = range(1, 21)

    for k in k_values:
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed)
        gmm.fit(x)
        aic_scores.append(gmm.aic(x))
        bic_scores.append(gmm.bic(x))

    # Plot AIC vs. k
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, aic_scores, marker="o")
    plt.title("GMM AIC")
    plt.xlabel("Number of components (k)")
    plt.ylabel("AIC Score")
    plt.show(block=False)

    # Plot BIC vs. k
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, bic_scores, marker="o")
    plt.title("GMM BIC")
    plt.xlabel("Number of components (k)")
    plt.ylabel("BIC Score")
    plt.show(block=False)

    # Determine aic_elbow_k and bic_elbow_k manually from the plots
    aic_elbow_k = int(input("\nEnter the aic_elbow_k value from the AIC plot: "))
    bic_elbow_k = int(input("Enter the bic_elbow_k observed from the BIC plot: "))

    # Use predict() method for k=aic_elbow_k
    gmm_aic = GaussianMixture(
        n_components=aic_elbow_k, covariance_type="diag", random_state=seed
    )
    labels_aic = gmm_aic.fit_predict(x)

    if aic_elbow_k == 3:
        matched_labels_aic = match_labels(y, labels_aic)
        cm_aic = confusion_matrix(y, matched_labels_aic)
        acc_aic = accuracy_score(y, matched_labels_aic)
        print("\nConfusion Matrix for aic_elbow_k:")
        print(cm_aic)
        print("Accuracy for aic_elbow_k:", acc_aic)
    else:
        print(
            "\nCannot calculate Accuracy Score because the number of classes is not the same as the number of clusters."
        )
        cm_aic = confusion_matrix(y, labels_aic)
        print("Confusion Matrix for aic_elbow_k:")
        print(cm_aic)

    # Use predict() method for k=bic_elbow_k
    gmm_bic = GaussianMixture(
        n_components=bic_elbow_k, covariance_type="diag", random_state=seed
    )
    labels_bic = gmm_bic.fit_predict(x)

    if bic_elbow_k == 3:
        matched_labels_bic = match_labels(y, labels_bic)
        cm_bic = confusion_matrix(y, matched_labels_bic)
        acc_bic = accuracy_score(y, matched_labels_bic)
        print("\nConfusion Matrix for bic_elbow_k:")
        print(cm_bic)
        print("Accuracy for bic_elbow_k:", acc_bic)
    else:
        print(
            "\nCannot calculate Accuracy Score because the number of classes is not the same as the number of clusters."
        )
        cm_bic = confusion_matrix(y, labels_bic)
        print("Confusion Matrix for bic_elbow_k:")
        print(cm_bic)


def part3(seed: int, x, y) -> None:
    # Part 3: Self-Organizing Map
    print("\nPart 3:")

    # Normalize the features
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)

    # Initialize and train SOMs with different grid sizes
    grid_sizes = [3, 7, 15, 25]
    quantization_errors = []

    for size in grid_sizes:
        som = MiniSom(
            size,
            size,
            x_normalized.shape[1],
            sigma=1.0,
            learning_rate=0.5,
            random_seed=seed,
        )
        som.train_random(x_normalized, 1000)

        q_error = som.quantization_error(x_normalized)
        quantization_errors.append(q_error)

        # Plotting the U-Matrix and response
        plt.figure(figsize=(7, 7))
        plt.pcolor(
            som.distance_map().T, cmap="bone_r"
        )  # plotting the distance map as background
        plt.colorbar()
        markers = ["o", "s", "D"]
        colors = ["r", "g", "b"]
        for cnt, xx in enumerate(x_normalized):
            w = som.winner(xx)
            plt.plot(
                w[0] + 0.5,
                w[1] + 0.5,
                markers[y[cnt]],
                markersize=8,
                markerfacecolor="None",
                markeredgecolor=colors[y[cnt]],
                markeredgewidth=2,
            )
        plt.title(f"SOM Grid Size: {size}x{size}")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    # Plot quantization error vs. grid sizes
    plt.figure(figsize=(8, 4))
    plt.plot(grid_sizes, quantization_errors, marker="o")
    plt.title("Quantization Error vs Grid Size")
    plt.xlabel("Grid Size")
    plt.ylabel("Quantization Error")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

    for size, q_error in zip(grid_sizes, quantization_errors):
        print(f"Grid Size {size}x{size}: Quantization Error = {q_error}")


def run(seed: int = 658) -> None:
    # Load the iris dataset
    iris = load_iris()
    x = iris.data
    y = iris.target

    part1(seed, x, y)
    part2(seed, x, y)
    part3(seed, x, y)


if __name__ == "__main__":
    run()
