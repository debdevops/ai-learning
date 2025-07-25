# Week 2: Supervised Learning & Linear Regression

### Objective
To understand the fundamentals of supervised machine learning and build our first predictive models. We will teach a machine to predict a continuous value (regression) by learning the relationship between a set of input features and the target variable.

### Key Concepts

1.  **Supervised Learning:**
    *   The most common category of machine learning. It is "supervised" because we train the model on a dataset that contains both the **input features (X)** and the **correct answers (y)**, also known as labels or the target. The model's goal is to learn a mapping function `f(X) = y`.

2.  **Regression vs. Classification:**
    *   **Regression:** A task where the goal is to predict a continuous numerical value. *Example: Predicting the price of a house.*
    *   **Classification:** A task where the goal is to predict a discrete category. *Example: Predicting if an email is 'spam' or 'not spam'.*
    *   This week focuses entirely on **regression**.

3.  **Exploratory Data Analysis (EDA) with Seaborn:**
    *   Before modeling, we must understand our data. A `pairplot` is a powerful visualization tool that creates scatter plots between all variables in a dataset, allowing us to quickly spot potential linear relationships.

4.  **Linear Regression:**
    *   A fundamental regression algorithm that assumes a linear relationship between the input features and the target variable.
    *   **Simple Linear Regression:** Uses one feature to predict the target (`y = m*x + c`). The model learns a 2D line.
    *   **Multivariate Linear Regression:** Uses multiple features to predict the target (`y = m1*x1 + m2*x2 + ... + c`). The model learns a multi-dimensional hyperplane.
    *   **Model Coefficients (`m1`, `m2`...):** These are the weights learned by the model. After feature scaling, their magnitude indicates the importance and their sign indicates the direction (positive or negative) of each feature's impact on the target.

5.  **The Machine Learning Workflow:**
    *   **Data Splitting (`train_test_split`):** We divide our data into a **training set** and a **testing set**. The model learns from the training set and is evaluated on the unseen testing set to see how well it generalizes. This is critical to prevent the model from simply "memorizing" the data.
    *   **Feature Scaling (`StandardScaler`):** A crucial preprocessing step for multivariate models. It rescales features to have a similar range (e.g., a mean of 0 and a standard deviation of 1), preventing features with large numerical values from unfairly dominating the model.
    *   **Model Training (`.fit()`):** The process where the model analyzes the training data to learn the optimal coefficients.
    *   **Model Evaluation (`.score()`, MSE):** Quantifying the model's performance on the test data using metrics like R-squared (the proportion of the target's variance explained by the model) and Mean Squared Error (the average squared difference between predicted and actual values).

### Files in this folder
- `README.md`: This file, outlining the core theory for the week.
- `data/`: Folder containing our housing price and e-commerce datasets.
- `linear_regression_practice.py`: Implements a simple linear regression model with one feature (housing prices).
- `model_evaluation.py`: Demonstrates how to evaluate the simple model's performance on the test set, both quantitatively and visually.
- `multivariate_regression.py`: Implements a more advanced linear regression model using multiple features and introduces the critical concept of feature scaling (housing prices).
- `customer_spend_predictor.py`: A second, complete project reinforcing the multivariate regression workflow on a new e-commerce dataset. Introduces `seaborn` for exploratory data analysis and visualizes model performance by plotting actual vs. predicted values.