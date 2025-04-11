import matplotlib.pyplot as plt
import os

def plot_clusters(X, labels, title: str, output_path: str):
    """Plot clusters on 2D reduced space."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()