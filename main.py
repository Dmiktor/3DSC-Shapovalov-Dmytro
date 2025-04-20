import joblib
import os
import optuna
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from src.dimensionality import reduce_tsne, reduce_lle, reduce_umap, save_reduced
from src.clustering import cluster_kmeans, cluster_dbscan, cluster_agglomerative
from src.metrics import calculate_unsupervised_metrics
from src.visualization import plot_clusters
from src.metrics_writer import MetricsWriter

def optimize_kmeans(X, y):
    def objective(trial):
        n_clusters = trial.suggest_int('n_clusters', 2, 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        return silhouette

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_n_clusters = study.best_params['n_clusters']
    return best_n_clusters

def optimize_dbscan(X, y):
    def objective(trial):
        eps = trial.suggest_uniform('eps', 0.1, 1.0)
        min_samples = trial.suggest_int('min_samples', 2, 10)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        # If the number of clusters is 1, the silhouette score is not valid, so we avoid it.
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            return silhouette
        else:
            return -1.0  # Return a low score if the clustering is not valid.

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_eps = study.best_params['eps']
    best_min_samples = study.best_params['min_samples']
    return best_eps, best_min_samples

def optimize_agglomerative(X, y):
    def objective(trial):
        n_clusters = trial.suggest_int('n_clusters', 2, 10)
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        return silhouette

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_n_clusters = study.best_params['n_clusters']
    return best_n_clusters

def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Load processed data
    X, y = joblib.load("processed/processed.pkl")

    # Reduce dimensionality
    X_tsne = reduce_tsne(X)
    X_lle = reduce_lle(X)
    X_umap = reduce_umap(X)

    save_reduced(X_tsne, "tsne", output_dir)
    save_reduced(X_lle, "lle", output_dir)
    save_reduced(X_umap, "umap", output_dir)

    reduced_data = {
        "tsne": X_tsne,
        "lle": X_lle,
        "umap": X_umap
    }

    best_kmeans_clusters = optimize_kmeans(X, y)
    best_dbscan_eps, best_dbscan_min_samples = optimize_dbscan(X, y)
    best_agglomerative_clusters = optimize_agglomerative(X, y)

    clustering_methods = {
        "kmeans": lambda X: cluster_kmeans(X, n_clusters=best_kmeans_clusters),
        "dbscan": lambda X: cluster_dbscan(X, eps=best_dbscan_eps, min_samples=best_dbscan_min_samples),
        "agglomerative": lambda X: cluster_agglomerative(X, n_clusters=best_agglomerative_clusters)
    }

    metrics_writer = MetricsWriter()

    for reduction_name, X_red in reduced_data.items():
        for cluster_name, cluster_func in clustering_methods.items():
            print(f"Processing {reduction_name} + {cluster_name}")

            labels_pred = cluster_func(X_red)

            silhouette, db_score, ari = calculate_unsupervised_metrics(X_red, labels_pred, y)

            metrics = {
                "Silhouette Score": silhouette if silhouette is not None else "N/A",
                "Davies-Bouldin Score": db_score if db_score is not None else "N/A",
                "Adjusted Rand Index": ari if ari is not None else "N/A"
            }

            metrics_writer.save_metrics(metrics, reduction_name, cluster_name)
            if np.all(labels_pred == -1):
                print(f"{reduction_name} + {cluster_name}: No clusters found (all -1). Skipping plot.")
            continue
            plot_clusters(
                X_red,
                labels_pred,
                title=f"{reduction_name} + {cluster_name}",
                output_path=os.path.join(output_dir, f"{reduction_name}_{cluster_name}.png")
            )

if __name__ == "__main__":
    main()