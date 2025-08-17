# --- Step 0: Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # <-- This is our new unsupervised model

print("Libraries imported.")

# --- Step 1: Load and Prepare the Data ---
file_path = './data/mall_customers.csv'
df = pd.read_csv(file_path)

print("\n--- Mall Customer Data ---")
print(df.head())
df.info()

# For this analysis, we want to segment customers based on their income and spending.
# Let's select the 'Annual_Income_k' and 'Spending_Score' columns.
# We will use .values to get a NumPy array, which is what the model expects.
X = df[['Annual_Income_k', 'Spending_Score']].values

# Let's visualize our data first to see if we can spot any natural groups.
print("\nVisualizing the raw customer data...")
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1])
plt.title('Raw Mall Customer Data')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show() # Close the plot window to continue


# --- Step 2: Use the Elbow Method to Find the Optimal Number of Clusters (K) ---
# We don't know the "right" number of customer segments. Is it 3? 4? 5?
# The Elbow Method helps us choose a good value for K.

print("\nRunning the Elbow Method to find the best K...")
# WCSS stands for "Within-Cluster Sum of Squares". It's a measure of how
# tightly packed the points are within a cluster. A smaller WCSS is better.
wcss = []

# We will test K from 1 to 10.
for i in range(1, 11):
    # Create a KMeans instance for the given number of clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    # Fit the model to our data X
    kmeans.fit(X)
    # Append the WCSS value to our list
    wcss.append(kmeans.inertia_)

# Now, we plot the WCSS for each K.
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show() # Close the plot window to continue

print("Elbow Method analysis complete.")

# --- Step 3: Train the Final K-Means Model with the Optimal K ---
# Based on our Elbow Method plot, the optimal number of clusters is K=5.
optimal_k = 5

# Create the final KMeans model with 5 clusters.
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)

# Fit the model to the data and predict the cluster for each data point.
# For clustering, .fit_predict() is a convenient shortcut that does both steps at once.
y_kmeans = kmeans_final.fit_predict(X)

print(f"\nModel training with K={optimal_k} complete.")
print("Each data point has now been assigned to a cluster.")
# You can uncomment the line below to see the cluster assignment for each customer
# print(y_kmeans)


# --- Step 4: Visualize the Final Clusters ---
# This is the most important visualization. We will create a scatter plot,
# but this time we'll color each point according to the cluster it belongs to.
print("\nVisualizing the final customer segments...")
plt.figure(figsize=(12, 8))

# We will create a scatter plot for each cluster, one by one.
# Cluster 0: Customers where y_kmeans is 0
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1 (Careful)')
# Cluster 1: Customers where y_kmeans is 1
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2 (Standard)')
# Cluster 2: Customers where y_kmeans is 2
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3 (Target)')
# Cluster 3: Customers where y_kmeans is 3
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4 (Careless)')
# Cluster 4: Customers where y_kmeans is 4
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5 (Sensible)')

# We also plot the centroids of each cluster, the "center of gravity".
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', edgecolors='black')

plt.title('Clusters of Mall Customers', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

print("\nClustering analysis complete.")