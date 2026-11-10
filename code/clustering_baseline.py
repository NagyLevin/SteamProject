
import pandas as pd

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

X = df[features]
#print(X.size)
# Standardize the behavioral features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train K-Means
kmeans = KMeans(
    n_clusters=3, #number of clusters to create
    max_iter=300, #max number of iterations to find the best clusters
    random_state=13,
    n_init=10 #number of times to run the clustering algorithm with different initial centroids
)

kmeans.fit(X_scaled) #every user is assigned to a cluster

# Get cluster labels
labels = kmeans.labels_ #groups of users

#print(labels)

# Evaluate clustering
silhouette = silhouette_score(
    X_scaled,
    labels,
    sample_size=6000, #to reduce computation time use only a sample
    random_state=13
)

dbi = davies_bouldin_score(X_scaled, labels)

print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Index:", round(dbi, 4))
print("Inertia:", round(kmeans.inertia_, 2))

# Display cluster sizes
df["cluster"] = labels

print("Cluster Sizes:")
print(df["cluster"].value_counts().sort_index())
