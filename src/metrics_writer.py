import os
import matplotlib.pyplot as plt
import pandas as pd

class MetricsWriter:
    def __init__(self):
        pass
    
    def save_metrics(self, metrics, reduction_name, cluster_name):
        method_dir = f"results/{reduction_name}_{cluster_name}"
        
        os.makedirs(method_dir, exist_ok=True)
        
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        metrics_df.to_csv(f"{method_dir}/metrics.csv", index=False)
        
        self.plot_metrics(metrics, method_dir)

    def plot_metrics(self, metrics, method_dir):
        metric_names = list(metrics.keys())
        metric_values = [float(value) if isinstance(value, (int, float)) else 0.0 for value in metrics.values()]

        plt.bar(metric_names, metric_values, color='skyblue')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Clustering Metrics')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{method_dir}/metrics_plot.png")
        plt.close()