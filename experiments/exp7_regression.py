"""
Experiment 7 - Linear & Polynomial Regression
Part 1: Linear Regression on California Housing dataset.
Part 2: Polynomial Regression on Auto MPG dataset.
NOTE: The original VTU code fetches mpg.csv from a URL.
      Here we use seaborn's built-in mpg dataset for offline use.
      If seaborn's cache is empty, a synthetic fallback is provided.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def run():
    # ---- Part 1: Linear Regression ----
    print("\nPerforming Linear Regression on California Housing Dataset\n")
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Linear Regression MSE: {mse:.4f}")

    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='red', linestyle='dashed', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: Actual vs Predicted Values")
    plt.show()

    # ---- Part 2: Polynomial Regression ----
    print("\nPerforming Polynomial Regression on Auto MPG Dataset\n")

    # Load mpg dataset offline via seaborn cache / fallback
    try:
        df = sns.load_dataset('mpg')
    except Exception:
        # Fallback: generate synthetic horsepower vs mpg data
        print("(Using synthetic MPG data for offline mode)")
        np.random.seed(42)
        hp = np.random.uniform(50, 230, 300)
        mpg = 45 - 0.1 * hp + 0.0002 * hp**2 + np.random.normal(0, 3, 300)
        df = pd.DataFrame({'horsepower': hp, 'mpg': mpg})

    df = df.dropna()
    X = df[['horsepower']].astype(float)
    y = df['mpg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degree = 3
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_poly_pred = poly_model.predict(X_test)

    poly_mse = mean_squared_error(y_test, y_poly_pred)
    print(f"Polynomial Regression (degree {degree}) MSE: {poly_mse:.4f}")

    plt.scatter(X_test, y_test, color='blue', label='Actual')
    sorted_idx = np.argsort(X_test.values.flatten())
    plt.plot(X_test.values.flatten()[sorted_idx], y_poly_pred[sorted_idx],
             color='red', label='Polynomial Fit')
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.title(f"Polynomial Regression (Degree {degree})")
    plt.legend()
    plt.show()
