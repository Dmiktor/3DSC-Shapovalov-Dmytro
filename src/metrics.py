from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import numpy as np

def calculate_unsupervised_metrics(X, labels_pred, labels_true=None):
    unique_labels = np.unique(labels_pred)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        silhouette = None
        db_score = None
    else:
        silhouette = silhouette_score(X, labels_pred)
        db_score = davies_bouldin_score(X, labels_pred)

    if labels_true is not None:
        ari = adjusted_rand_score(labels_true, labels_pred)
    else:
        ari = None

    return silhouette, db_score, ari