"""
Experiment 3 - PCA
Uses Iris dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def run():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target

    print("Iris Dataset Overview:")
    print(data.head())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, :-1])

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_data, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['species'] = data['species']

    print("\nExplained Variance Ratio by Each Principal Component:")
    print(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='species',
        palette='Set1',
        data=pca_df,
        s=100
    )
    plt.title('PCA: Iris Dataset (2D Projection)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Species')
    plt.grid(alpha=0.3)
    plt.show()
