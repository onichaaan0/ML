"""
Experiment 1 - EDA Histogram & Box Plot
Uses California Housing dataset.
"""

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


def run():
    # Step 2: Load the California Housing Dataset
    california_data = fetch_california_housing()
    data = pd.DataFrame(california_data.data,
                         columns=california_data.feature_names)
    data['MedHouseVal'] = california_data.target

    # Step 3: Create Histograms for Numerical Features
    plt.figure(figsize=(15, 12))
    for i, column in enumerate(data.columns):
        plt.subplot(4, 3, i + 1)
        plt.hist(data[column], bins=30, color='skyblue',
                 edgecolor='black')
        plt.title(f'Histogram of {column}')
    plt.tight_layout()
    plt.show()

    # Step 4: Generate Box Plots for Numerical Features
    plt.figure(figsize=(15, 12))
    for i, column in enumerate(data.columns):
        plt.subplot(4, 3, i + 1)
        sns.boxplot(y=data[column], color='lightcoral')
        plt.title(f'Box Plot of {column}')
    plt.tight_layout()
    plt.show()

    # Step 5: Analyze the Visualizations
    print("Analyze the histograms and box plots to observe distributions and outliers.")

    # Step 6: Compute and Display Outliers Using IQR
    def compute_outliers(dataframe):
        print("\nOutlier Detection Using IQR Method:")
        for column in dataframe.columns:
            Q1 = dataframe[column].quantile(0.25)
            Q3 = dataframe[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = dataframe[(dataframe[column] < lower_bound) |
                                  (dataframe[column] > upper_bound)]
            print(f"\nFeature: {column}")
            print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
            print(f"Number of Outliers: {len(outliers)}")
            if not outliers.empty:
                print(f"Outlier Values:\n{outliers[column].values}")

    compute_outliers(data)
