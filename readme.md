README for Sales and Rock Classification Project

Overview

This project addresses multiple data processing and machine learning tasks using Python. The tasks involved predicting sales values and classifying rocks into sandstone and shale, using datasets provided in CSV and other formats. Below is a breakdown of the questions posed and the steps taken to solve them.

Questions Addressed

1. Predict sales from the sales transactions dataset

Objective:

To predict the dependent variable (TotalSalesValue) using the sales transaction dataset.

Approach:

Dataset: sales-transactions-dataset

Preprocessing:

Handled missing values.

Encoded categorical variables using LabelEncoder.

Split the dataset into training and testing sets.

Model Used: Linear Regression.

Steps:

Used the LinearRegression class from sklearn to fit a model on the training data.

Predicted TotalSalesValue for the testing set.

Evaluated model performance using metrics like R-squared.

Outcome:

A regression model capable of predicting sales values was created and its performance was evaluated.

2. Classify rocks into sandstone and shale from the rock dataset

Objective:

To classify rocks as sandstone or shale based on specific features.

Approach:

Dataset: classification-for-sandstone-and-shale

Preprocessing:

Handled missing values and standardized the dataset.

Encoded the target variable (Class(Shale=1,SS=0)) for binary classification.

Split the dataset into training and testing sets.

Model Used: Logistic Regression.

Steps:

Used the LogisticRegression class from sklearn to fit a model on the training data.

Predicted the rock type (sandstone or shale) for the testing set.

Evaluated model performance using metrics like accuracy and confusion matrix.

Outcome:

A classification model was built to successfully classify rocks with evaluated performance metrics.

General Steps for Each Dataset

Data Extraction:

Extracted datasets from .zip files using Python's zipfile library.

Preprocessing:

Handled missing values.

Encoded categorical data.

Standardized numeric data where necessary.

Model Building:

Selected appropriate models for the tasks (Linear Regression for prediction, Logistic Regression for classification).

Split the data into training and testing sets.

Model Evaluation:

Used appropriate metrics for regression (e.g., R-squared) and classification (e.g., accuracy).

Predictions:

Saved predictions to CSV files for future reference.

Key Files

Python Scripts:

zip.py: Main script for extracting datasets and preprocessing data.

linear_regression.py: Script for predicting sales using linear regression.

logistic_regression.py: Script for classifying rocks using logistic regression.

Datasets:

For_Prediction_Test.csv: Processed test data for sales prediction.

For_Prediction_Data1.csv: Processed data for rock classification.

Libraries Used

pandas: Data manipulation and preprocessing.

numpy: Numerical computations.

scikit-learn: Machine learning algorithms and metrics.

Results and Observations

Linear regression provided predictions for sales values with a measured R-squared value.

Logistic regression classified rocks into sandstone and shale, achieving a high accuracy score.

Future Work

Explore advanced models like decision trees or random forests for potentially better predictions.

Enhance preprocessing by feature engineering and outlier detection.

Author

Itulua Cheluh Osedebhamie
