"""
Experiment 10 - K-Means Clustering
Uses Breast Cancer dataset (first 2 features).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run():
    data = load_breast_cancer()
    X = data.data[:, :2]
    feature_names = data.feature_names[:2]
    y_true = data.target

    print(f"Dataset shape: {X.shape}")
    print(f"Features used: {list(feature_names)}")

    # Apply K-Means with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    print(f"\nCluster Centers:\n{centers}")
    print(f"\nSilhouette Score: {silhouette_score(X, y_pred):.4f}")

    # Plot the clustering result
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1],
                c='blue', label='Cluster 0', alpha=0.6)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1],
                c='green', label='Cluster 1', alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1],
                c='red', marker='X', s=200, label='Centroids')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('K-Means Clustering (Breast Cancer Dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compare with actual labels
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1],
                c='red', label='Malignant', alpha=0.6)
    plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1],
                c='blue', label='Benign', alpha=0.6)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Actual Labels (Breast Cancer Dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()
