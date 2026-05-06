"""
ML LAB REFERENCE CLI
Main entry point. Shows each experiment's source code, runs it,
and displays the output.
"""

import inspect

# Import each experiment module from the experiments package
from experiments import (
    pca,
    knn,
    finds,
    decision_tree,
    naive_bayes,
    kmeans,
    regression,
    lwr,
    eda_histogram,
    correlation,
)


def show_code(module):
    """Print the full source code of the given experiment module."""
    print("\n============= CODE =============\n")
    print(inspect.getsource(module))
    print("\n===============================\n")


def show_menu():
    """Print the main menu to the terminal."""
    print("\n========================")
    print(" ML LAB REFERENCE CLI")
    print("========================\n")
    print("1. PCA")
    print("2. KNN")
    print("3. Find-S")
    print("4. Decision Tree")
    print("5. Naive Bayes")
    print("6. K-Means")
    print("7. Regression")
    print("8. Locally Weighted Regression")
    print("9. EDA Histogram")
    print("10. Correlation Matrix")
    print("0. Exit")


def main():
    # Map menu options to (label, module)
    options = {
        "1": ("PCA", pca),
        "2": ("KNN", knn),
        "3": ("Find-S", finds),
        "4": ("Decision Tree", decision_tree),
        "5": ("Naive Bayes", naive_bayes),
        "6": ("K-Means", kmeans),
        "7": ("Regression", regression),
        "8": ("Locally Weighted Regression", lwr),
        "9": ("EDA Histogram", eda_histogram),
        "10": ("Correlation Matrix", correlation),
    }

    # Run an interactive loop until the user chooses to exit
    while True:
        show_menu()
        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("Exiting. Goodbye!")
            break

        if choice in options:
            name, module = options[choice]
            # Step 1: show the source code of the experiment
            show_code(module)
            # Step 2: run the experiment
            print("============= OUTPUT =============\n")
            try:
                module.run()
            except Exception as e:
                # Catch errors so the menu does not crash
                print(f"Error while running {name}: {e}")
            print("\n==================================\n")
            input("Press Enter to return to the menu...")
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
