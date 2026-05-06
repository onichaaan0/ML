"""
Experiment 4 - Find-S Algorithm
Uses in-memory dataset (EnjoySport) instead of external CSV for offline use.
Original VTU code used pd.read_csv(); adapted here with embedded data.
"""

import pandas as pd


def run():
    def find_s_algorithm(data):
        features = data.iloc[:, :-1].values
        target = data.iloc[:, -1].values

        hypothesis = None

        for i, row in enumerate(features):
            if target[i] == 'Yes':
                hypothesis = list(row)
                break

        if hypothesis is None:
            print("No positive examples found in the data.")
            return None

        for i, row in enumerate(features):
            if target[i] == 'Yes':
                for j in range(len(hypothesis)):
                    if hypothesis[j] != row[j]:
                        hypothesis[j] = '?'

        return hypothesis

    # In-memory dataset (replaces pd.read_csv for offline use)
    data = pd.DataFrame({
        'Sky':      ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
        'AirTemp':  ['Warm',  'Warm',  'Cold',  'Warm'],
        'Humidity': ['Normal','High',  'High',  'High'],
        'Wind':     ['Strong','Strong','Strong','Strong'],
        'Water':    ['Warm',  'Warm',  'Warm',  'Cool'],
        'Forecast': ['Same',  'Same',  'Change','Change'],
        'EnjoySport':['Yes',  'Yes',   'No',    'Yes'],
    })

    print("Training Data:")
    print(data)

    hypothesis = find_s_algorithm(data)

    if hypothesis is not None:
        print("\nThe most specific hypothesis consistent with the training examples is:")
        print(hypothesis)
