
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
        f"Adjusted Rand Index: {ari:.4f}"
    )
