
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
from pathlib import Path

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



# Visualize K=4 cluster on grafikonm
cluster_sizes = df["cluster_k4"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values)

plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("Steam Player Segments - Log K-Means (K=4)")

for i, count in enumerate(cluster_sizes.values):
    plt.text(i, count + 50, str(count), ha="center")

# Save figure
output_path = (
    Path(__file__).resolve().parent.parent # cd .. cd.. 
    / "figures"
    / "log_k4_cluster_sizes.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

# Test clustering stability with different random states

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("K=4 Clustering Stability Test")

random_states = [0, 7, 13, 24, 42, 99]

for seed in random_states:
    model = KMeans(
        n_clusters=4,
        max_iter=300,
        random_state=seed,
        n_init=10
    )

    new_labels = model.fit_predict(X_scaled)

    # Compare with our original K=4 model
    ari = adjusted_rand_score(labels4, new_labels)

    print(
        f"Random State: {seed} | "
        f"Adjusted Rand Index: {ari:.4f}"
    )

# Select features for visualization
comparison_features = [
    "games_purchased",
    "total_playtime",
    "play_ratio"
]

# Get cluster averages that i need using original data
cluster_comparison = (
    df.groupby("cluster_k4")[comparison_features]
    .mean()
)

# Create three charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = [
    "Average Games Purchased",
    "Average Total Playtime",
    "Average Play Ratio"
]

# Plot each feature
for i, feature in enumerate(comparison_features):
    values = cluster_comparison[feature]

    axes[i].bar(
        values.index.astype(str),
        values.values
    )

    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel(feature.replace("_", " ").title())

    if feature == "play_ratio":
        axes[i].set_ylim(0, 1.1)

    # Display values above the bars
    for j, value in enumerate(values):
        axes[i].text(
            j,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

plt.suptitle("Steam Player Behavior by Cluster (K=4)")
plt.tight_layout()

# Save figure
output_path = (
    Path(__file__).resolve().parent.parent
    / "figures"
    / "log_k4_feature_comparison.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()