# Week 3: Logistic Regression for Classification

### Objective
To understand and implement our first classification algorithm. We will move from predicting continuous values (regression) to predicting discrete categories. Our goal is to build a model that can predict whether a user will purchase a product based on their age and salary.

### Key Concepts

1.  **Classification:**
    *   A type of supervised learning where the goal is to predict a categorical label. This is different from regression, where we predict a numerical value.
    *   **Binary Classification:** A task with only two possible outcomes. Examples: Yes/No, True/False, Spam/Not Spam, Purchased/Not Purchased. This is our focus for this week.

2.  **The Sigmoid (Logistic) Function:**
    *   The core mathematical component of Logistic Regression. It's an "S"-shaped curve that takes any real-valued number and maps it to a value between 0 and 1. This output is interpreted as a **probability**.

3.  **Decision Boundary & Threshold:**
    *   The model learns a line or curve (the decision boundary) that best separates the classes.
    *   Based on which side of the boundary a data point falls, the model calculates a probability. We use a **threshold** (typically 0.5) to convert this probability into a final class prediction (e.g., if probability > 0.5, predict class 1).

4.  **New Evaluation Metrics for Classification:**
    *   Metrics like R-squared and MSE don't work for classification. We need new tools.
    *   **Accuracy:** The simplest metric. It's the percentage of predictions the model got correct. (Accuracy = Correct Predictions / Total Predictions).
    *   **Confusion Matrix:** A table that gives a more detailed breakdown of performance, showing us where the model got things right and where it got confused (e.g., correctly predicting "Yes," incorrectly predicting "Yes," etc.).

### Files in this folder
- `README.md`: This file.
- `data/social_network_ads.csv`: A dataset of user information and whether they purchased a product.
- `logistic_regression_practice.py`: A script to build, train, and evaluate our first classification model.