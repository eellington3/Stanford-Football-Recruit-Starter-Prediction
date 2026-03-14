# CS-141-Sports-and-Data

# CS 141 Sports and Data — Final Project

## Overview
This repository contains the code used for the final project in **CS 141: Sports and Data**. The project analyzes college football recruiting data to evaluate whether recruiting information can predict **early player contribution**, specifically whether a recruit becomes a starter within the first two seasons.

The script `copy_of_finalcs141.py` implements the main data processing, modeling, and evaluation pipeline used in the project. It loads player data, prepares features, trains several machine learning models, and evaluates their performance on predicting early career outcomes.

---

# File Description

## `copy_of_finalcs141.py`
This is the main Python script for the project. It performs the following steps:

### 1. Data Loading
- Reads the processed dataset containing recruit attributes and early-career performance metrics.
- Uses **pandas** for data management and preprocessing.

### 2. Feature Engineering
- Selects relevant player characteristics such as:
  - Recruiting rating
  - Position
  - Physical attributes (height and weight)
  - Conference
- Converts categorical variables into numerical form using **one-hot encoding**.

### 3. Data Splitting
- Splits the dataset into **training and testing sets** to evaluate model performance on unseen data.

### 4. Model Training
The script trains several machine learning models to predict whether a player becomes an early starter:

- Logistic Regression
- Random Forest
- XGBoost
- Stacked ensemble model

### 5. Model Evaluation
Models are evaluated using several classification metrics:

- Precision
- Recall
- F1 Score
- ROC curves

Confusion matrices are also generated to examine prediction errors.

### 6. Feature Importance Analysis
The script analyzes which variables contribute most to predictions using:

- Permutation importance
- SHAP value analysis (for tree-based models)

### 7. Visualization
Plots are generated to help interpret model performance and feature importance.


