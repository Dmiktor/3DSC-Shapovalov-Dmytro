from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def cluster_kmeans(X, n_clusters=2, random_state=42):
    """Cluster data using KMeans."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    return model.fit_predict(X)

def cluster_dbscan(X, eps=0.5, min_samples=5):
    """Cluster data using DBSCAN."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def cluster_agglomerative(X, n_clusters=2):
    """Cluster data using Agglomerative Clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X)