from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import numpy as np

def calculate_unsupervised_metrics(X, labels_pred, labels_true=None):
    
    labels_pred = np.array(labels_pred)
    
    if len(np.unique(labels_pred)) < 2 or np.all(labels_pred == -1):
        silhouette = None
        db_score = None
    else:
        try:
            silhouette = silhouette_score(X, labels_pred)
        except:
            silhouette = None

        try:
            db_score = davies_bouldin_score(X, labels_pred)
        except:
            db_score = None

    if labels_true is not None:
        try:
            ari = adjusted_rand_score(labels_true, labels_pred)
        except:
            ari = None
    else:
        ari = None

    return silhouette, db_score, ari