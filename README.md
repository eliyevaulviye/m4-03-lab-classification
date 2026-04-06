![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Classification

## Overview

Classification is one of the most common tasks in machine learning — from spam detection to species identification, the ability to assign data points to the correct category is a core skill for any data professional. But choosing the right classification algorithm isn't always straightforward, and the "best" model depends heavily on the data and the problem.

In this lab, you'll work with the Palmer Penguins dataset to train and compare multiple classification algorithms. The dataset contains physical measurements and categorical attributes of three penguin species observed in Antarctica — a modern, accessible dataset that includes both numerical and categorical features. You'll go beyond simple accuracy to evaluate models using precision, recall, F1 score, confusion matrices, and ROC curves — the metrics that matter in real-world applications where misclassification has consequences.

By the end, you'll have a clear comparison framework that you can reuse in any classification project, and you'll understand which algorithms tend to work well (and why).

## Learning Goals

By the end of this lab, you should be able to:

- Train and evaluate a logistic regression baseline with accuracy, precision, recall, and F1.
- Compare multiple classification algorithms on the same dataset.
- Interpret confusion matrices and ROC curves to assess model quality.
- Use GridSearchCV to tune hyperparameters and improve model performance.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. All analysis, code, and written interpretations should live in a single notebook so that your reasoning is visible alongside the output.

This lab applies the classification techniques from today's lesson. You'll use scikit-learn's classification estimators and model selection tools, along with matplotlib and seaborn for plotting ROC curves and confusion matrices.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Getting Started

1. Create a new Jupyter Notebook called **`m4-03-classification.ipynb`**.
2. Start with an import cell:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             ConfusionMatrixDisplay, classification_report)

sns.set_style("whitegrid")
```

3. Work through the tasks in order. Each task builds on the previous one.
4. Include markdown cells between code cells to explain your observations and reasoning.

## Tasks

### Task 1: Data Prep & Baseline

Load the dataset and establish a performance baseline.

1. Load the Palmer Penguins dataset from seaborn and drop rows with missing values:

```python
penguins = sns.load_dataset("penguins").dropna()
```

2. Explore briefly: how many samples and features? What are the three species and their distribution? Which columns are categorical vs. numerical?
3. Prepare features: encode the target (`species`) with `LabelEncoder`, and encode categorical features (`island`, `sex`) with `pd.get_dummies(drop_first=True)`.
4. Split into training and test sets (80/20, `stratify=y`, `random_state=42`).
5. Scale the features using `StandardScaler` (fit on train, transform both).
6. Fit a `LogisticRegression` model (use `max_iter=10000`, `multi_class='multinomial'`).
7. Report **accuracy**, **precision**, **recall**, and **F1 score** on the test set. Print a full `classification_report`.
8. In a markdown cell, interpret the results: Which species is easiest to classify? Which is hardest? Why might that be?

### Task 2: Algorithm Comparison

Train multiple classifiers and compare their performance.

1. Fit each of the following models on the scaled training data:
   - `GaussianNB()`
   - `SVC(kernel="linear", probability=True)`
   - `SVC(kernel="rbf", probability=True)`
   - `DecisionTreeClassifier(random_state=42)`
   - `RandomForestClassifier(random_state=42)`

2. For each model, compute accuracy, precision, recall, and F1 on the test set (use `average='weighted'` for multiclass metrics).
3. Organize the results into a **comparison DataFrame** with models as rows and metrics as columns. Sort by F1 score descending.
4. In a markdown cell, discuss: Which models perform best? Are there any surprises? Why might some algorithms outperform others on this dataset?

### Task 3: Confusion Matrices & ROC Curves

Visualize how each model makes its decisions.

1. **Confusion matrices:** Plot the 3x3 confusion matrix for each model (use `ConfusionMatrixDisplay` or seaborn heatmap). Arrange them in a grid of subplots for easy comparison. For each, note which species pairs are most frequently confused.

2. **ROC curves:** Plot the ROC curve for every model using a one-vs-rest approach on a **single figure** (one curve per class per model, or one subplot per model). Include the AUC value in each legend entry. Add a diagonal dashed line representing a random classifier.

3. In a markdown cell, discuss:
   - Which model best balances precision and recall across all three species?
   - Which species pair is hardest for the models to distinguish? Why might this be the case given the features?
   - Based on the confusion matrices and ROC curves, which model would you recommend?

### Task 4: Hyperparameter Exploration

Tune the best-performing model from Task 2.

1. Select the best model based on your Task 2 results.
2. Define a hyperparameter grid with **at least 3 hyperparameters** to tune. For example, if RandomForest was best:

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10]
}
```

3. Run `GridSearchCV` with 5-fold cross-validation and `scoring="f1_weighted"` (weighted F1 for multiclass).
4. Report the **best parameters** and the **best cross-validation F1 score**.
5. Evaluate the tuned model on the test set. Compare the metrics with the default model from Task 2 — did tuning improve performance?
6. In a markdown cell, reflect: Was the improvement significant? Is there a risk of overfitting to the validation folds? When is hyperparameter tuning most impactful?

## Submission

### What to submit

- `m4-03-classification.ipynb` — your completed notebook with all code, outputs, and markdown explanations.

### Definition of done (checklist)

- [ ] Penguins dataset is loaded, explored, and split with stratification.
- [ ] Categorical features are encoded and the target is label-encoded.
- [ ] Logistic regression baseline is trained and evaluated with four metrics.
- [ ] Five additional classifiers are trained and compared in a summary table.
- [ ] Confusion matrices are plotted for all models.
- [ ] ROC curves are plotted with AUC values.
- [ ] GridSearchCV is used to tune the best model with at least 3 hyperparameters.
- [ ] Markdown cells explain reasoning, especially around species confusion and metric trade-offs.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete classification models comparison"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
