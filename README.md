# ML & Data Science Projects

This repository contains a collection of machine learning, data mining, and data science projects completed as part of my academic coursework and personal learning. The projects focus on applying Python-based data analysis, preprocessing, machine learning modeling, evaluation, and interpretation techniques to real-world and publicly available datasets.

## Overview

The goal of this repository is to demonstrate my ability to work through the full data science workflow, including:

- Understanding the problem and dataset
- Cleaning and preprocessing structured data
- Performing exploratory data analysis
- Building machine learning pipelines
- Training and comparing multiple models
- Evaluating models using appropriate metrics
- Interpreting model performance and feature importance
- Communicating results through reports, scripts, and visualizations

The repository is intended to showcase practical experience with supervised learning, classification, clustering, feature engineering, model evaluation, and data-driven problem solving.

## Projects

### 1. Obesity Level Classification Using Machine Learning

**Course:** CPS 844 - Data Mining  
**Date:** Jan 2026 - Apr 2026  
**Type:** Multi-class classification  

This project predicts obesity levels based on demographic characteristics, eating habits, and physical-condition features. The dataset comes from the UCI Machine Learning Repository and contains 2,111 records with 16 predictive attributes and one target label representing seven obesity categories.

The project compares several machine learning classifiers, including:

- Logistic Regression
- Linear Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting

The best-performing model was **Gradient Boosting**, which achieved approximately **95% accuracy** and a **0.951 Weighted-F1 score** on the holdout test set.

#### Key Skills Demonstrated

- Multi-class classification
- Data preprocessing with `Pipeline` and `ColumnTransformer`
- Numeric and categorical feature handling
- Missing-value imputation
- Standardization and one-hot encoding
- Stratified train-test splitting
- Model comparison using Accuracy, Macro-F1, and Weighted-F1
- Confusion matrix and classification report generation
- Permutation feature importance
- Debugging Scikit-Learn version and preprocessing issues

#### Main Files

```text
ObesityDataSet_raw_and_data_sinthetic.csv
script_EmmaGogic.py
report_EmmaGogic.pdf
outputs/
