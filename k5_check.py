import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

from config import Config


def run_kmeans_k5():
    K = 5  # Override config, force K=5

    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING (K=5 EXPERIMENT)")
    print("=" * 70)

    # Create separate output directories for K=5
    k5_features_dir = os.path.join(Config.FEATURES_DIR, 'k5')
    k5_plots_dir = os.path.join(Config.PLOTS_DIR, 'k5')
    k5_csv_dir = os.path.join(Config.CSV_DIR, 'k5')

    os.makedirs(k5_features_dir, exist_ok=True)
    os.makedirs(k5_plots_dir, exist_ok=True)
    os.makedirs(k5_csv_dir, exist_ok=True)

    print(f"\nOutputs will be saved to:")
    print(f"   Features: {k5_features_dir}")
    print(f"   Plots: {k5_plots_dir}")
    print(f"   CSV: {k5_csv_dir}")

    print("\n1. Loading training features...")

    features_path = os.path.join(Config.FEATURES_DIR, 'train_features.npy')
    labels_path = os.path.join(Config.FEATURES_DIR, 'train_labels.npy')
    filenames_path = os.path.join(Config.FEATURES_DIR, 'train_filenames.npy')

    train_features = np.load(features_path)
    train_labels = np.load(labels_path)
    train_filenames = np.load(filenames_path, allow_pickle=True)

    n_samples, n_features = train_features.shape
    print(f"   Loaded {n_samples} images")
    print(f"   Each image has {n_features} features (latent dimensions)")

    print("\n2. Standardizing features...")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(train_features)

    print(f"   Before: mean={train_features.mean():.4f}, std={train_features.std():.4f}")
    print(f"   After:  mean={features_scaled.mean():.4f}, std={features_scaled.std():.4f}")

    # Save scaler for evaluation
    scaler_path = os.path.join(k5_features_dir, 'scaler_params.npz')
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"   Saved scaler to: {scaler_path}")

    print(f"\n3. Running K-Means with K={K} clusters...")

    kmeans = KMeans(
        n_clusters=K,
        random_state=Config.KMEANS_RANDOM_STATE,
        max_iter=Config.KMEANS_MAX_ITER,
        n_init=Config.KMEANS_N_INIT
    )

    cluster_labels = kmeans.fit_predict(features_scaled)

    print(f"   Done! Converged in {kmeans.n_iter_} iterations")
    print(f"   Final inertia: {kmeans.inertia_:.2f}")
    print(f"   Cluster centers shape: ({K}, {n_features})")

    # Save cluster centers
    centers_path = os.path.join(k5_features_dir, 'cluster_centers.npy')
    np.save(centers_path, kmeans.cluster_centers_)
    print(f"   Saved centers to: {centers_path}")

    print(f"\n4. Cluster distribution:")

    for i in range(K):
        count = (cluster_labels == i).sum()
        pct = count / n_samples * 100
        print(f"   Cluster {i}: {count:5d} images ({pct:5.1f}%)")

    print("\n5. Measuring clustering quality...")

    silhouette = silhouette_score(features_scaled, cluster_labels)
    print(f"   Silhouette Score: {silhouette:.4f}")
    print(f"   (Range: -1 to 1, higher = better separated clusters)")

    print("\n6. Creating visualization...")

    pca = PCA(n_components=2, random_state=Config.RANDOM_SEED)
    features_2d = pca.fit_transform(features_scaled)

    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"   PCA keeps {variance_explained:.1f}% of variance in 2D")

    # Cluster scatter plot
    plt.figure(figsize=(12, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i in range(K):
        mask = cluster_labels == i
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[i],
            label=f'Cluster {i} (n={mask.sum()})',
            alpha=0.5,
            s=30
        )

    # Plot cluster centers
    centers_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='black',
        marker='X',
        s=200,
        edgecolors='white',
        linewidths=2,
        label='Cluster Centers'
    )

    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.title(f'K-Means Clustering (K=5)\nSilhouette Score: {silhouette:.4f}',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(k5_plots_dir, 'kmeans_clusters_k5.png')
    plt.savefig(plot_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path}")

    # Cluster sizes bar chart
    plt.figure(figsize=(10, 6))
    cluster_counts = [np.sum(cluster_labels == i) for i in range(K)]
    bars = plt.bar(range(K), cluster_counts, color=colors, edgecolor='black')

    for bar, count in zip(bars, cluster_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{count}', ha='center', fontsize=11, fontweight='bold')

    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Cluster Sizes (K=5)', fontsize=14, fontweight='bold')
    plt.xticks(range(K))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    sizes_path = os.path.join(k5_plots_dir, 'cluster_sizes_k5.png')
    plt.savefig(sizes_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {sizes_path}")

    print("\n7. Saving cluster assignments...")

    assignments_path = os.path.join(k5_features_dir, 'train_cluster_assignments.npy')
    np.save(assignments_path, cluster_labels)
    print(f"   Saved: {assignments_path}")

    # Save CSV
    csv_path = os.path.join(k5_csv_dir, 'train_cluster_assignments_k5.csv')
    with open(csv_path, 'w') as f:
        f.write('filename,true_label,cluster\n')
        for fname, label, cluster in zip(train_filenames, train_labels, cluster_labels):
            f.write(f'{fname},{label},{cluster}\n')
    print(f"   Saved: {csv_path}")

    print("\n8. Saving clustering report...")

    report_path = os.path.join(k5_features_dir, 'clustering_report_k5.txt')
    with open(report_path, 'w') as f:
        f.write("K-MEANS CLUSTERING REPORT (K=5)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of clusters: {K}\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Feature dimensions: {n_features}\n")
        f.write(f"Silhouette Score: {silhouette:.4f}\n")
        f.write(f"Inertia: {kmeans.inertia_:.2f}\n")
        f.write(f"Iterations: {kmeans.n_iter_}\n\n")
        f.write("Cluster Distribution:\n")
        for i in range(K):
            count = (cluster_labels == i).sum()
            pct = count / n_samples * 100
            f.write(f"   Cluster {i}: {count} images ({pct:.1f}%)\n")
    print(f"   Saved: {report_path}")

    print("\n" + "=" * 70)
    print("K=5 CLUSTERING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   {n_samples} images grouped into {K} clusters")
    print(f"   Silhouette Score: {silhouette:.4f}")
    print(f"\n📁 Files saved to:")
    print(f"   {k5_features_dir}")
    print(f"   {k5_plots_dir}")
    print(f"   {k5_csv_dir}")
    print(f"\n🚀 Next step: Run evaluation_k5.py")


if __name__ == "__main__":
    run_kmeans_k5()