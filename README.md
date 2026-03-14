# CS-141-Sports-and-Data

Overview

This repository contains the code used for the final project in CS 141: Sports and Data, a course that introduces data analysis and machine learning techniques using sports applications as the motivating domain.

The script copy_of_finalcs141.py implements the main data processing, modeling, and evaluation pipeline used in the project. The goal of the project is to analyze college football recruiting data and evaluate whether recruiting information can predict early player contribution, such as becoming a starter within the first two seasons.

The script loads player data, prepares features, trains several machine learning models, and evaluates their performance on predicting early career outcomes.

File Description
copy_of_finalcs141.py

This is the main Python script for the project. It performs the following steps:

Data Loading

Reads in the processed dataset containing recruit attributes and early-career performance metrics.

Uses pandas to manage and clean the data.

Feature Engineering

Selects relevant player characteristics such as:

Recruiting rating

Position

Physical attributes (height and weight)

Conference

Converts categorical variables into numerical form using one-hot encoding.

Data Splitting

Splits the dataset into training and testing sets to evaluate model performance on unseen data.

Model Training

Several machine learning models are trained to predict early player contribution:

Logistic Regression

Random Forest

XGBoost

Stacked ensemble model

Model Evaluation

Models are evaluated using classification metrics including:

Precision

Recall

F1 score

ROC curves

Confusion matrices are used to examine prediction errors.

Feature Importance Analysis

The script also analyzes which variables contribute most to predictions using:

Permutation importance

SHAP value analysis (for tree-based models)

Visualization

Generates plots to help interpret model performance and feature effects.
