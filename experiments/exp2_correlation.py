"""
Experiment 2 - Correlation Matrix & Pair Plot
Uses California Housing dataset.
"""

from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def run():
    CAL_data = fetch_california_housing(as_frame=True)
    data = CAL_data.frame

    correlation_matrix = data.corr()
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    sns.pairplot(data, diag_kind='kde')
    plt.show()
