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

### 2. Retail Association Analysis and Wholesale Customer Clustering

**Course:** CPS 844 - Data Mining  
**Date:** Mar 2026  
**Type:** Unsupervised learning, association-rule mining, and clustering  

This project applies two unsupervised data mining techniques to business/customer datasets from the UCI Machine Learning Repository. The first part uses the **Online Retail** dataset to perform market-basket analysis and discover product co-purchase patterns. The second part uses the **Wholesale Customers** dataset to segment customers based on annual spending behavior.

The project demonstrates how different unsupervised learning methods can answer different business questions:

- **Association analysis:** Which products are frequently purchased together?
- **Clustering analysis:** Which customers have similar spending patterns?

#### Association Analysis

For the Online Retail dataset, each invoice was treated as a customer basket and each product description was treated as an item. The data was cleaned by removing missing invoice/product values, excluding cancelled transactions, filtering invalid quantities and prices, standardizing product descriptions, and restricting the analysis to United Kingdom transactions for a cleaner basket structure.

The project used the **FP-Growth algorithm** to mine frequent itemsets and generate association rules based on support, confidence, and lift.

Key settings included:

- Top 120 products by basket frequency
- Minimum support: 0.02
- Minimum confidence: 0.30
- Minimum lift: 1.10
- Maximum itemset length: 3

The final analysis produced **309 frequent itemsets** and **345 association rules**. The strongest rules showed that customers often purchased coordinated product variations together, such as different CHARLOTTE BAG designs and REGENCY teacup-and-saucer variants.

#### Clustering Analysis

For the Wholesale Customers dataset, the project used six annual spending features:

- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

The `Region` and `Channel` columns were excluded from clustering so that customer groups could be discovered based only on purchasing behavior.

Since the spending variables were highly skewed, the data was preprocessed using:

- `log1p` transformation
- Z-score standardization

Two clustering methods were compared:

- K-Means clustering
- Agglomerative hierarchical clustering

K-Means was evaluated for `k = 2` to `k = 8` using inertia and silhouette score. The best K-Means solution was **k = 2**, with a silhouette score of **0.2903**. Agglomerative clustering with two clusters achieved a slightly lower silhouette score of **0.2585**.

The final clusters represented two broad customer segments:

1. Customers with higher spending on **Fresh** and **Frozen** products
2. Customers with higher spending on **Milk**, **Grocery**, and **Detergents_Paper**

#### Key Skills Demonstrated

- Unsupervised learning
- Market-basket analysis
- Association-rule mining
- FP-Growth algorithm
- Support, confidence, and lift interpretation
- Customer segmentation
- K-Means clustering
- Agglomerative hierarchical clustering
- Silhouette score and inertia analysis
- PCA-based cluster visualization
- Hierarchical clustering dendrograms
- Data cleaning and preprocessing
- Business interpretation of data mining results

#### Main Files

```text
Online Retail.xlsx
Wholesale customers data.csv
script_EmmaGogic.py
report_EmmaGogic.pdf
assignment2_outputs/
