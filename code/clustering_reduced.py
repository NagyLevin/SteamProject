
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
df = pd.read_csv("data/processed/user_features.csv")

# Select reduced feature set
features = [
    "games_purchased",
    "games_played",
    "total_playtime",
    "play_ratio"
]

X = df[features].copy()

# Apply log transformation
log_features = [
    "games_purchased",
    "games_played",
    "total_playtime"
]

X[log_features] = np.log1p(X[log_features])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means with K=4
kmeans = KMeans(
    n_clusters=4,
    max_iter=300,
    random_state=13,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

# Evaluate model
silhouette = silhouette_score(
    X_scaled,
    labels,
    sample_size=6000,
    random_state=13
)

dbi = davies_bouldin_score(X_scaled, labels)

print("Reduced Feature Model Results:")
print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Index:", round(dbi, 4))
print("Inertia:", round(kmeans.inertia_, 2))

# Display cluster sizes
df["cluster"] = labels

print("Cluster Sizes:")
print(df["cluster"].value_counts().sort_index())

# Display cluster profiles using all original features
original_features = [
    "games_purchased",
    "games_played",
    "total_playtime",
    "avg_playtime",
    "max_playtime",
    "play_ratio"
]

print("Cluster Averages:")
print(
    df.groupby("cluster")[original_features]
    .mean()
    .round(2)
    .to_string()
)

print("Cluster Medians:")
print(
    df.groupby("cluster")[original_features]
    .median()
    .round(2)
    .to_string()
)

print("Reduced K=4 Stability Test")

random_states = [1, 7, 13, 24, 42, 99]

for seed in random_states:
    model = KMeans(
        n_clusters=4,
        max_iter=300,
        random_state=seed,
        n_init=10
    )

    new_labels = model.fit_predict(X_scaled)

    ari = adjusted_rand_score(labels, new_labels)

    print(
        f"Random State: {seed} | "
        f"Adjusted Rand Index: {ari:.4f}" #round up to 4
    )

#Visualise final modell

cluster_sizes = df["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
bars = plt.bar(
    cluster_sizes.index.astype(str),
    cluster_sizes.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("Steam Player Segments - Reduced K-Means (K=4)")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 50,
        f"{int(height)}",
        ha="center"
    )

plt.ylim(0, cluster_sizes.max() * 1.12)
plt.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent
    / "figures"
    / "reduced_k4_cluster_sizes.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()



# Compare behavioral features of users in the final model # 
comparison_features = [
    "games_purchased",
    "total_playtime",
    "play_ratio"
]

cluster_comparison = (
    df.groupby("cluster")[comparison_features]
    .mean()
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = [
    "Average Games Purchased",
    "Average Total Playtime",
    "Average Play Ratio"
]

for i, feature in enumerate(comparison_features):
    values = cluster_comparison[feature]

    bars = axes[i].bar(
        values.index.astype(str),
        values.values
    )

    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel(feature.replace("_", " ").title())

    if feature == "play_ratio":
        axes[i].set_ylim(0, 1.1)
    else:
        axes[i].set_ylim(0, values.max() * 1.15)

    for bar, value in zip(bars, values):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

plt.suptitle("Steam Player Behavior - Reduced K-Means (K=4)")
plt.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent
    / "figures"
    / "reduced_k4_feature_comparison.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

