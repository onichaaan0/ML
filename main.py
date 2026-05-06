from experiments import (
    exp1_eda,
    exp2_correlation,
    exp3_pca,
    exp4_finds,
    exp5_knn,
    exp6_lwr,
    exp7_regression,
    exp8_decision_tree,
    exp9_naive_bayes,
    exp10_kmeans
)

import inspect


def show_code(module):

    print("\n============= SOURCE CODE =============\n")

    print(inspect.getsource(module))

    print("\n=======================================\n")


def menu():

    print("\n========================")
    print(" ML LAB REFERENCE CLI")
    print("========================")

    print("1. Experiment 1 - EDA Histogram & Boxplot")
    print("2. Experiment 2 - Correlation Matrix")
    print("3. Experiment 3 - PCA")
    print("4. Experiment 4 - Find-S")
    print("5. Experiment 5 - KNN")
    print("6. Experiment 6 - LWR")
    print("7. Experiment 7 - Regression")
    print("8. Experiment 8 - Decision Tree")
    print("9. Experiment 9 - Naive Bayes")
    print("10. Experiment 10 - KMeans")
    print("0. Exit")


while True:

    menu()

    choice = input("\nEnter your choice: ")

    if choice == "1":

        show_code(exp1_eda)
        exp1_eda.run()

    elif choice == "2":

        show_code(exp2_correlation)
        exp2_correlation.run()

    elif choice == "3":

        show_code(exp3_pca)
        exp3_pca.run()

    elif choice == "4":

        show_code(exp4_finds)
        exp4_finds.run()

    elif choice == "5":

        show_code(exp5_knn)
        exp5_knn.run()

    elif choice == "6":

        show_code(exp6_lwr)
        exp6_lwr.run()

    elif choice == "7":

        show_code(exp7_regression)
        exp7_regression.run()

    elif choice == "8":

        show_code(exp8_decision_tree)
        exp8_decision_tree.run()

    elif choice == "9":

        show_code(exp9_naive_bayes)
        exp9_naive_bayes.run()

    elif choice == "10":

        show_code(exp10_kmeans)
        exp10_kmeans.run()

    elif choice == "0":

        print("Exiting...")
        break

    else:

        print("Invalid Choice")