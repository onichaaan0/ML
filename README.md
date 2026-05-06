# ML LAB REFERENCE CLI

An **offline**, terminal-based Machine Learning lab reference for VTU ML Lab experiments.
Built entirely in Python using `scikit-learn`, `numpy`, `pandas`, `matplotlib`, and `seaborn`.

No web app, no database, no internet required after installation.

## Project Structure

```
ML-LAB-CLI/
├── main.py
├── experiments/
│   ├── pca.py
│   ├── knn.py
│   ├── finds.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   ├── kmeans.py
│   ├── regression.py
│   ├── lwr.py
│   ├── eda_histogram.py
│   └── correlation.py
├── theory/
├── outputs/
├── requirements.txt
└── README.md
```

## Experiments

1. **PCA** - Iris dataset, 4D -> 2D with `StandardScaler`, scatter plot.
2. **KNN** - 100 random numbers, label first 50, predict last 50.
3. **Find-S** - Manual implementation, most specific hypothesis.
4. **Decision Tree** - Breast Cancer dataset, accuracy.
5. **Naive Bayes** - Olivetti Faces dataset, GaussianNB accuracy.
6. **K-Means** - Breast Cancer (first 2 features), cluster plot.
7. **Regression** - California Housing, Linear Regression MSE.
8. **Locally Weighted Regression** - Synthetic sine wave with LWR curve.
9. **EDA Histogram** - Histograms for California Housing columns.
10. **Correlation Matrix** - Seaborn heatmap on California Housing.

## Installation

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

## Running the project

From the project root:

```bash
python main.py
```

You will see a menu like:

```
========================
 ML LAB REFERENCE CLI
========================

1. PCA
2. KNN
3. Find-S
4. Decision Tree
5. Naive Bayes
6. K-Means
7. Regression
8. Locally Weighted Regression
9. EDA Histogram
10. Correlation Matrix
0. Exit
```

Type the number of the experiment you want to run and press Enter.

> Note: The Olivetti Faces dataset (Experiment 5) is downloaded once
> by scikit-learn and cached locally. After the first run it works
> completely offline.

## Building a Standalone EXE (Windows)

You can package the CLI into a single executable using PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile main.py
```

The generated executable will be inside the `dist/` folder:

```
dist/main.exe
```

Run it directly:

```bash
dist\main.exe
```

## Folders

- `experiments/` - one Python file per experiment, each exposing `run()`.
- `theory/` - place your notes / theory references here.
- `outputs/` - place generated plots / output files here if desired.
