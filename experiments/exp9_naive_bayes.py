"""
Experiment 9 - Naive Bayes
Uses Olivetti Faces dataset with GaussianNB.
NOTE: fetch_olivetti_faces downloads once on first run, then works offline.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


def run():
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    print(f"Shape of data (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")
    print(f"Number of unique classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy of the Naive Bayes Classifier: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=[f"Class {i}" for i in np.unique(y)]))

    def visualize_predictions(X, y_true, y_pred, indices):
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(indices):
            plt.subplot(2, len(indices) // 2, i + 1)
            plt.imshow(X[idx].reshape(64, 64), cmap='gray')
            plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    random_indices = np.random.choice(len(y_test), size=8, replace=False)
    visualize_predictions(X_test, y_test, y_pred, random_indices)
