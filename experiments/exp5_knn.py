"""
Experiment 5 - KNN
Generates 100 random numbers, labels first 50, predicts remaining 50.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def run():
    np.random.seed(42)
    x_values = np.random.rand(100)

    labels = np.array(['Class1' if x <= 0.5 else 'Class2' for x in x_values[:50]])

    X_train = x_values[:50].reshape(-1, 1)
    y_train = labels
    X_test = x_values[50:].reshape(-1, 1)

    k_values = [1, 2, 3, 4, 5, 20, 30]

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        print(f"\nResults for k = {k}:")
        for i, (x, pred) in enumerate(zip(x_values[50:], predictions), start=51):
            print(f"x{i} = {x:.3f}, Predicted Class = {pred}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, [1]*len(X_train),
                c=['blue' if label == 'Class1' else 'red' for label in y_train],
                label='Training Points')
    plt.scatter(X_test, [0]*len(X_test), c='green', label='Test Points')
    plt.axvline(0.5, color='gray', linestyle='--', label='Boundary x=0.5')
    plt.yticks([], [])
    plt.legend(loc='upper center')
    plt.title("Training and Test Points")
    plt.xlabel("x values")
    plt.show()
