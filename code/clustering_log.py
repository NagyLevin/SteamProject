
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load user-level dataset
df = pd.read_csv("data/processed/user_features.csv")

# Select behavioral features
features = [
    "games_purchased",
    "games_played",
    "total_playtime",
    "avg_playtime",
    "max_playtime",
    "play_ratio"
]

X = df[features].copy()

# Apply logarithmic transformation
log_features = [
    "games_purchased",
    "games_played",
    "total_playtime",
    "avg_playtime",
    "max_playtime"
]

X[log_features] = np.log1p(X[log_features]) #apply logarithmic transformation to the features like discussed in the EDA 

# Standardize the transformed features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train K-Means
kmeans = KMeans(
    n_clusters=3, #use the same number of clusters as the baseline model
    max_iter=300, #use the same number of iterations as the baseline model
    random_state=13, #use the same random state as the baseline model
    n_init=10 #use the same number of initializations as the baseline model
)

kmeans.fit(X_scaled) #train the model

# Get cluster labels
labels = kmeans.labels_ #get the cluster labels

# Evaluate clustering
silhouette = silhouette_score(
    X_scaled,
    labels,
    sample_size=6000,
    random_state=13
)

dbi = davies_bouldin_score(X_scaled, labels)

print("Log Transformation Results:")
print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Index:", round(dbi, 4))
print("Inertia:", round(kmeans.inertia_, 2))

# Display cluster sizes
df["cluster"] = labels

print("Cluster Sizes:")
print(df["cluster"].value_counts().sort_index())

# Calculate cluster averages
cluster_means = df.groupby("cluster")[features].mean()

print("Cluster Averages:")
print(cluster_means.round(2).to_string())

# Calculate cluster medians
cluster_medians = df.groupby("cluster")[features].median()

print("Cluster Medians:")
print(cluster_medians.round(2).to_string())


# Compare different K values after log transformation
print("Log K-Means Model Comparison")

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        max_iter=300,
        random_state=13,
        n_init=10
    )

    labels_k = model.fit_predict(X_scaled)

    silhouette_k = silhouette_score(
        X_scaled,
        labels_k,
        sample_size=6000,
        random_state=13
    )

    dbi_k = davies_bouldin_score(X_scaled, labels_k)

    cluster_sizes = pd.Series(labels_k).value_counts()
    largest_cluster_pct = cluster_sizes.max() / len(labels_k) * 100

    print(
        f"K={k} | "
        f"Silhouette: {silhouette_k:.4f} | "
        f"Davies-Bouldin: {dbi_k:.4f} | "
        f"Largest Cluster: {largest_cluster_pct:.1f}%"
    )



# Train the selected K=4 model
kmeans4 = KMeans(
    n_clusters=4,
    max_iter=300,
    random_state=13,
    n_init=10
)

labels4 = kmeans4.fit_predict(X_scaled)

# Add new cluster labels to the original dataset
df["cluster_k4"] = labels4

# Display cluster sizes
print("K=4 Cluster Sizes:")
print(df["cluster_k4"].value_counts().sort_index())

# Display average values
cluster_means4 = df.groupby("cluster_k4")[features].mean()

print("K=4 Cluster Averages:")
print(cluster_means4.round(2).to_string())

# Display median values
cluster_medians4 = df.groupby("cluster_k4")[features].median()

print("K=4 Cluster Medians:")
print(cluster_medians4.round(2).to_string())

#TODO make a grafikon mybe to visualise data