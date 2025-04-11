from sklearn.manifold import TSNE, LocallyLinearEmbedding
import umap
import joblib
import os

def reduce_tsne(X, n_components=2, random_state=42):
    """Reduce dimensionality using t-SNE."""
    model = TSNE(n_components=n_components, random_state=random_state)
    return model.fit_transform(X)

def reduce_lle(X, n_components=2, n_neighbors=10, random_state=42):
    """Reduce dimensionality using Locally Linear Embedding."""
    model = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
        eigen_solver='dense'
    )
    return model.fit_transform(X)

def reduce_umap(X, n_components=2, random_state=42):
    """Reduce dimensionality using UMAP."""
    model = umap.UMAP(n_components=n_components, random_state=random_state)
    return model.fit_transform(X)

def save_reduced(X_reduced, name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(X_reduced, os.path.join(output_dir, f"{name}.pkl"))