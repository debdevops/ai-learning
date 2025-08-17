# Week 5: Unsupervised Learning - K-Means Clustering

### Objective
To explore our first Unsupervised Learning algorithm. Unlike previous weeks, we will work with data that has no "correct" answer or target variable. Our goal is to use the K-Means algorithm to identify natural groupings or "clusters" of customers based on their shopping behavior.

### Key Concepts

1.  **Unsupervised Learning:**
    *   A category of machine learning where the model is given data without explicit labels (`y`). The goal is not to predict an outcome, but to find hidden patterns, structures, and relationships within the data itself.

2.  **Clustering:**
    *   The most common unsupervised task. The objective is to partition data points into a number of distinct groups (clusters) where points within a single group are very similar to each other, and points in different groups are dissimilar.

3.  **K-Means Algorithm:**
    *   An iterative algorithm that aims to find a user-specified number of clusters (`K`).
    *   **Centroids:** The center point of a cluster.
    *   **How it works:** It randomly initializes `K` centroids, assigns each data point to the nearest centroid, recalculates the new centroid for each cluster, and repeats this process until the clusters stabilize.

4.  **The Elbow Method:**
    *   A heuristic used to help determine the optimal number of clusters (`K`). We run the K-Means algorithm for a range of `K` values and plot the "within-cluster sum of squares" (WCSS) for each. The "elbow" of the resulting curve, where the rate of decrease sharply changes, is a good estimate for the best `K`.

### Files in this folder
- `README.md`: This file.
- `data/mall_customers.csv`: A dataset containing information on mall shoppers' income and spending habits.
- `kmeans_clustering.py`: A script that performs K-Means clustering to segment the mall customers and visualizes the resulting clusters.