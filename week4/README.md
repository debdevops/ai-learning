# Week 4: Tree-Based Models - Decision Trees & Random Forests

### Objective
To move beyond linear models and explore non-linear, tree-based algorithms for classification. This week, we learn how to build an intuitive Decision Tree and then see how to improve upon its main weakness by using an ensemble method called a Random Forest.

### Key Concepts

1.  **Decision Tree:**
    *   A supervised learning model that works like a flowchart. It partitions the data into subsets based on a series of "if/then" questions related to the feature values.
    *   **Intuitive & Interpretable:** Its greatest strength is that you can visualize the tree and easily understand the exact rules the model has learned.
    *   **Main Weakness (Overfitting):** A single, deep tree can learn the training data *too* perfectly, including its noise and specific quirks. This causes it to perform poorly on new, unseen data, as it has not learned the general underlying patterns.

2.  **Ensemble Learning:**
    *   The core idea that combining many simple or weak models can create one powerful and robust model. This is a "wisdom of the crowd" approach. The Random Forest is a prime example of this.

3.  **Random Forest:**
    *   An **ensemble** of many Decision Trees.
    *   **How it works:** It builds hundreds of individual trees. Each tree is trained on a random sample of the training data and considers only a random subset of features for its splits.
    *   **Prediction via Voting:** To classify a new data point, it gets a prediction from every tree in the forest. The final prediction is the class that receives the most "votes."
    *   **Main Strength:** By averaging the predictions of many diverse trees, it dramatically reduces overfitting, leading to higher accuracy and better generalization on unseen data.

4.  **Data Preparation for Trees (One-Hot Encoding):**
    *   Tree-based models, like linear models, require all input features to be numerical. We practiced using the `pd.get_dummies()` function to convert categorical text data (like 'job' or 'marital' status) into a numerical format (one-hot encoding) that the model can understand.

### Files in this folder
- `README.md`: This file, outlining the core theory for tree-based models.
- `data/bank_marketing.csv`: A dataset used to predict whether a customer will subscribe to a term deposit.
- `decision_tree_practice.py`: A complete script that cleans the data and then builds, trains, and evaluates both a single Decision Tree and a more powerful Random Forest model, allowing for direct comparison.