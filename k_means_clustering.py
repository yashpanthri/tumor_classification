import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
from datetime import datetime
import csv

from config import Config


def load_features(dataset_name='train'):
    features_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_features.npy')
    labels_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_labels.npy')
    filenames_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_filenames.npy')

    if not os.path.exists(features_path):
        print(f"Error: Features not found at {features_path}")
        print("Run extract_features.py first!")
        return None, None, None

    features = np.load(features_path)
    labels = np.load(labels_path)
    filenames = np.load(filenames_path, allow_pickle=True)

    return features, labels, filenames


def run_kmeans_simple():

    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING")
    print("=" * 70)

    # Create output directories
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.CSV_DIR, exist_ok=True)
    os.makedirs(Config.FEATURES_DIR, exist_ok=True)

    print("\n1. Loading training features...")
    train_features, train_labels, train_filenames = load_features('train')
    if train_features is None:
        return

    print(f"   Loaded {train_features.shape[0]} images")
    print(f"   Each image has {train_features.shape[1]} features (latent dimensions)")

    print("\n2. Standardizing features...")
    print("   (Makes all features have mean=0, std=1 so they contribute equally)")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(train_features)

    print(f"   Before: mean={train_features.mean():.4f}, std={train_features.std():.4f}")
    print(f"   After:  mean={features_scaled.mean():.4f}, std={features_scaled.std():.4f}")

    # Save scaler for later use on test/validate
    scaler_path = os.path.join(Config.FEATURES_DIR, 'scaler_params.npz')
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"   Saved scaler to: {scaler_path}")

    print(f"\n3. Running K-Means with K={Config.N_CLUSTERS} clusters...")
    print("   How it works:")
    print("   - Start with 4 random center points")
    print("   - Assign each image to nearest center")
    print("   - Move centers to mean of assigned images")
    print("   - Repeat until centers stop moving")

    kmeans = KMeans(
        n_clusters=Config.N_CLUSTERS,
        random_state=Config.KMEANS_RANDOM_STATE,
        max_iter=Config.KMEANS_MAX_ITER,
        n_init=Config.KMEANS_N_INIT  # Run 10 times, keep best
    )

    cluster_labels = kmeans.fit_predict(features_scaled)

    print(f"\n   Done! Converged in {kmeans.n_iter_} iterations")
    print(f"   Final inertia (sum of distances to centers): {kmeans.inertia_:.2f}")

    # Cluster centers (these are in 128D space)
    cluster_centers = kmeans.cluster_centers_
    print(f"   Cluster centers shape: {cluster_centers.shape}")

    # Save cluster centers
    centers_path = os.path.join(Config.FEATURES_DIR, 'cluster_centers.npy')
    np.save(centers_path, cluster_centers)
    print(f"   Saved centers to: {centers_path}")

    print("\n4. Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for c, n in zip(unique, counts):
        pct = n / len(cluster_labels) * 100
        print(f"   Cluster {c}: {n:4d} images ({pct:5.1f}%)")

    print("\n5. Measuring clustering quality...")
    silhouette = silhouette_score(features_scaled, cluster_labels)
    print(f"   Silhouette Score: {silhouette:.4f}")
    print("   (Range: -1 to 1, higher = better separated clusters)")

    if silhouette > 0.5:
        print("   Interpretation: Strong cluster structure")
    elif silhouette > 0.25:
        print("   Interpretation: Reasonable cluster structure")
    else:
        print("   Interpretation: Weak cluster structure (clusters overlap)")

    print("\n6. Creating visualization...")
    print("   Note: Features are 128D, can't plot directly")
    print("   Using PCA to reduce to 2D for visualization only")

    pca = PCA(n_components=2, random_state=Config.RANDOM_SEED)
    features_2d = pca.fit_transform(features_scaled)

    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"   PCA keeps {variance_explained:.1f}% of variance in 2D")

    # Also transform cluster centers to 2D for plotting
    centers_2d = pca.transform(cluster_centers)

    # Create scatter plot
    plt.figure(figsize=(12, 10))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']

    # Plot each cluster's points
    for cluster_id in range(Config.N_CLUSTERS):
        mask = cluster_labels == cluster_id
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[cluster_id],
            label=f'Cluster {cluster_id} (n={mask.sum()})',
            alpha=0.5,
            s=30
        )

    # Plot cluster centers as big X markers
    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='black',
        marker='X',
        s=300,
        edgecolors='white',
        linewidths=2,
        label='Cluster Centers',
        zorder=10
    )

    # Add center labels
    for i, (x, y) in enumerate(centers_2d):
        plt.annotate(
            f'C{i}',
            (x, y),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )

    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.title(f'K-Means Clustering Results\n(Silhouette Score: {silhouette:.3f})',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    cluster_plot_path = os.path.join(Config.PLOTS_DIR, 'kmeans_clusters.png')
    plt.savefig(cluster_plot_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {cluster_plot_path}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        unique, counts,
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(count),
            ha='center',
            fontsize=14,
            fontweight='bold'
        )

    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Images per Cluster', fontsize=14, fontweight='bold')
    plt.xticks(unique, [f'Cluster {c}' for c in unique])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    sizes_path = os.path.join(Config.PLOTS_DIR, 'cluster_sizes.png')
    plt.savefig(sizes_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {sizes_path}")

    print("\n7. Saving cluster assignments...")

    # Save as numpy array
    assignments_path = os.path.join(Config.FEATURES_DIR, 'train_cluster_assignments.npy')
    np.save(assignments_path, cluster_labels)
    print(f"   Saved: {assignments_path}")

    # Save as CSV (human readable)
    csv_path = os.path.join(Config.CSV_DIR, 'train_cluster_assignments.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'cluster', 'original_label'])
        for fname, cluster, label in zip(train_filenames, cluster_labels, train_labels):
            writer.writerow([fname, cluster, label])
    print(f"   Saved: {csv_path}")

    print("\n8. Saving clustering report...")

    report_path = os.path.join(Config.RESULTS_DIR, 'clustering_report.txt')
    with open(report_path, 'w') as f:
        f.write("K-MEANS CLUSTERING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Number of clusters (K): {Config.N_CLUSTERS}\n")
        f.write(f"  Latent dimension: {Config.LATENT_DIM}\n")
        f.write(f"  Random seed: {Config.KMEANS_RANDOM_STATE}\n\n")

        f.write("DATASET:\n")
        f.write(f"  Training images: {len(train_features)}\n")
        f.write(f"  Features per image: {train_features.shape[1]}\n\n")

        f.write("RESULTS:\n")
        f.write(f"  Iterations to converge: {kmeans.n_iter_}\n")
        f.write(f"  Inertia: {kmeans.inertia_:.2f}\n")
        f.write(f"  Silhouette Score: {silhouette:.4f}\n\n")

        f.write("CLUSTER DISTRIBUTION:\n")
        for c, n in zip(unique, counts):
            pct = n / len(cluster_labels) * 100
            f.write(f"  Cluster {c}: {n} images ({pct:.1f}%)\n")
        f.write("\n")

        f.write("VISUALIZATION:\n")
        f.write(f"  PCA variance explained: {variance_explained:.1f}%\n")

    print(f"   Saved: {report_path}")

    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)

    print(f"\nSummary:")
    print(f"   {len(train_features)} images grouped into {Config.N_CLUSTERS} clusters")
    print(f"   Silhouette Score: {silhouette:.4f}")

    print(f"\nFiles created:")
    print(f"   {cluster_plot_path}")
    print(f"   {sizes_path}")
    print(f"   {csv_path}")
    print(f"   {centers_path}")
    print(f"   {report_path}")

    print(f"\nNext step: Run evaluation.py to see how well clusters match tumor types")

    return {
        'kmeans': kmeans,
        'scaler': scaler,
        'cluster_labels': cluster_labels,
        'silhouette': silhouette
    }


if __name__ == "__main__":
    run_kmeans_simple()