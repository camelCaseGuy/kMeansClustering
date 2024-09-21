# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
# For this example, we are using the Mall Customer Segmentation Data from Kaggle.
# Dataset available at: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
data = pd.read_csv('Mall_Customers.csv')

# Display the first few rows of the dataset
print(data.head())

# Data preprocessing
# Drop irrelevant features like 'CustomerID', keep 'Annual Income' and 'Spending Score'
data_clean = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

# Convert the scaled data back to a DataFrame for better readability
data_scaled_df = pd.DataFrame(data_scaled, columns=['Annual Income', 'Spending Score'])

# Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(data_scaled_df)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# From the Elbow plot, we can choose the number of clusters as 5 (where the "elbow" occurs)

# Fit the KMeans model with the optimal hyper parameters
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(data_scaled_df)

# Add the cluster labels to the original DataFrame
data['Cluster'] = y_kmeans

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='Set1', s=100, legend="full")
plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0], kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
            s=300, c='yellow', label='Centroids', marker='x')
plt.title('Customer Segmentation (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Silhouette Score to evaluate clustering performance
sil_score = silhouette_score(data_scaled_df, y_kmeans)
print(f'Silhouette Score: {sil_score:.3f}')

# Display the data with cluster labels
print(data.head())
