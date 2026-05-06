"""
Experiment 6 - Locally Weighted Regression
Synthetic sine wave data with multiple tau values.
"""

import numpy as np
import matplotlib.pyplot as plt


def run():
    def locally_weighted_regression(x_query, X_train, y_train, tau):
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        x_query = np.array([1, x_query])
        weights = np.exp(-np.sum((X_train - x_query)**2, axis=1) / (2 * tau**2))
        W = np.diag(weights)
        theta = np.linalg.inv(X_train.T @ W @ X_train) @ (X_train.T @ W @ y_train)
        return x_query @ theta

    def plot_lwr(X, y, tau, resolution=100):
        X_test = np.linspace(X.min(), X.max(), resolution)
        y_pred = np.array([locally_weighted_regression(x_query, X, y, tau) for x_query in X_test])
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data Points')
        plt.plot(X_test, y_pred, color='red', label=f'LWR Curve (\u03c4={tau})')
        plt.title(f'Locally Weighted Regression (\u03c4={tau})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = np.sin(X) + np.random.normal(scale=0.2, size=X.shape)

    for tau in [0.1, 0.5, 1.0, 5.0]:
        plot_lwr(X, y, tau)
