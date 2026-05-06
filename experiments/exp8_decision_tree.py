"""
Experiment 8 - Decision Tree
Uses Breast Cancer dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


def run():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
    plt.show()

    new_sample = [X.iloc[0].tolist()]
    prediction = clf.predict(new_sample)
    print(f'Predicted class for new sample: {data.target_names[prediction[0]]}')
