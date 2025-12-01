import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import os
from datetime import datetime

from config import Config

# Binary class mapping
BINARY_NAMES = {0: 'No_Tumor', 1: 'Tumor'}


# Convert 4-class labels to binary
# glioma(0), meningioma(1), pituitary(2) → Tumor(1)
# no_tumor(3) → No_Tumor(0)
def to_binary(labels):
    binary = np.zeros_like(labels)
    binary[labels == 0] = 1  # glioma → Tumor
    binary[labels == 1] = 1  # meningioma → Tumor
    binary[labels == 2] = 1  # pituitary → Tumor
    binary[labels == 3] = 0  # no_tumor → No_Tumor
    return binary


def load_features(dataset_name):
    features_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_features.npy')
    labels_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_labels.npy')
    filenames_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_filenames.npy')

    if not os.path.exists(features_path):
        print(f"Error: Features not found at {features_path}")
        return None, None, None

    features = np.load(features_path)
    labels = np.load(labels_path)
    filenames = np.load(filenames_path, allow_pickle=True)

    return features, labels, filenames


def load_cluster_model():
    centers_path = os.path.join(Config.FEATURES_DIR, 'cluster_centers.npy')
    scaler_path = os.path.join(Config.FEATURES_DIR, 'scaler_params.npz')

    if not os.path.exists(centers_path):
        print(f"Error: Cluster centers not found at {centers_path}")
        print("Run k_means_clustering.py first!")
        return None, None

    cluster_centers = np.load(centers_path)

    scaler_params = np.load(scaler_path)
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']

    return cluster_centers, scaler


def assign_to_clusters(features, cluster_centers, scaler):
    features_scaled = scaler.transform(features)

    distances = np.zeros((len(features), len(cluster_centers)))
    for i, center in enumerate(cluster_centers):
        distances[:, i] = np.sqrt(np.sum((features_scaled - center) ** 2, axis=1))

    return np.argmin(distances, axis=1)


def run_binary_evaluation():
    print("\n" + "=" * 70)
    print("BINARY EVALUATION: TUMOR vs NO_TUMOR")
    print("(Using existing 4-class cluster assignments)")
    print("=" * 70)

    # Setup output directory
    eval_dir = os.path.join(Config.RESULTS_DIR, 'evaluation_binary')
    os.makedirs(eval_dir, exist_ok=True)
    print("\n1. Loading existing 4-class cluster model...")
    cluster_centers, scaler = load_cluster_model()
    if cluster_centers is None:
        return

    print(f"   Loaded {len(cluster_centers)} cluster centers")
    print(f"   Feature dimensions: {cluster_centers.shape[1]}")

    print("\n2. Loading test and validation features...")

    test_features, test_labels, test_filenames = load_features('test')
    val_features, val_labels, val_filenames = load_features('validate')

    if test_features is None or val_features is None:
        return

    print(f"   Test: {len(test_features)} images")
    print(f"   Validation: {len(val_features)} images")

    print("\n3. Assigning images to existing 4 clusters...")

    test_clusters = assign_to_clusters(test_features, cluster_centers, scaler)
    val_clusters = assign_to_clusters(val_features, cluster_centers, scaler)

    # Combine for evaluation
    all_labels = np.concatenate([test_labels, val_labels])
    all_clusters = np.concatenate([test_clusters, val_clusters])
    all_filenames = np.concatenate([test_filenames, val_filenames])

    print(f"   Total evaluation images: {len(all_labels)}")

    # Show 4-class cluster distribution
    print(f"\n   Cluster distribution:")
    for c in range(4):
        print(f"      Cluster {c}: {(all_clusters == c).sum()} images")

    print("\n4. Converting labels to binary...")

    all_binary_labels = to_binary(all_labels)

    tumor_count = (all_binary_labels == 1).sum()
    no_tumor_count = (all_binary_labels == 0).sum()

    print(f"   Tumor (glioma + meningioma + pituitary): {tumor_count}")
    print(f"   No_Tumor: {no_tumor_count}")

    print("\n5. Mapping clusters to binary using 4-class majority vote...")
    print("   (If cluster's majority 4-class label is no_tumor → No_Tumor)")
    print("   (If cluster's majority 4-class label is any tumor → Tumor)")

    cluster_to_binary = {}
    cluster_to_4class = {}  # Track the 4-class mapping too

    for cluster_id in range(4):
        mask = all_clusters == cluster_id
        labels_in_cluster = all_labels[mask]  # Original 4-class labels

        if len(labels_in_cluster) == 0:
            cluster_to_binary[cluster_id] = 1  # Default to Tumor
            cluster_to_4class[cluster_id] = -1
            continue

        # Find majority 4-class label (same as evaluation.py)
        unique_labels, counts = np.unique(labels_in_cluster, return_counts=True)
        majority_4class = unique_labels[np.argmax(counts)]
        majority_count = counts[np.argmax(counts)]
        total = len(labels_in_cluster)

        cluster_to_4class[cluster_id] = majority_4class

        # Map to binary: no_tumor (3) → No_Tumor (0), anything else → Tumor (1)
        if majority_4class == 3:  # no_tumor
            cluster_to_binary[cluster_id] = 0  # No_Tumor
            winner_binary = "No_Tumor"
        else:  # glioma, meningioma, or pituitary
            cluster_to_binary[cluster_id] = 1  # Tumor
            winner_binary = "Tumor"

        # Get 4-class name
        class_name = Config.CLASS_NAMES_4.get(majority_4class, f"Class {majority_4class}")

        # Also show binary breakdown for reference
        binary_labels = to_binary(labels_in_cluster)
        tumor_count = (binary_labels == 1).sum()
        no_tumor_count = (binary_labels == 0).sum()

        print(f"   Cluster {cluster_id}:")
        print(f"      4-class majority: {class_name} ({majority_count}/{total} = {majority_count / total * 100:.1f}%)")
        print(f"      Binary breakdown: Tumor={tumor_count}, No_Tumor={no_tumor_count}")
        print(f"      → Binary label: {winner_binary}")

    print(f"\n   Final binary mapping:")
    for c, binary_label in cluster_to_binary.items():
        class_4 = Config.CLASS_NAMES_4.get(cluster_to_4class[c], "Unknown")
        print(f"      Cluster {c} ({class_4}) = {BINARY_NAMES[binary_label]}")

    # Count how many clusters map to each binary class
    tumor_clusters = sum(1 for v in cluster_to_binary.values() if v == 1)
    no_tumor_clusters = sum(1 for v in cluster_to_binary.values() if v == 0)
    print(f"\n   {tumor_clusters} cluster(s) → Tumor, {no_tumor_clusters} cluster(s) → No_Tumor")

    print("\n6. Generating predictions...")

    predicted_binary = np.array([cluster_to_binary[c] for c in all_clusters])

    print("\n7. Calculating metrics...")

    accuracy = accuracy_score(all_binary_labels, predicted_binary)
    precision = precision_score(all_binary_labels, predicted_binary, zero_division=0)
    recall = recall_score(all_binary_labels, predicted_binary, zero_division=0)
    f1 = f1_score(all_binary_labels, predicted_binary, zero_division=0)

    print(f"\n   BINARY CLASSIFICATION RESULTS:")
    print(f"   ================================")
    print(f"   Accuracy:  {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1 Score:  {f1 * 100:.2f}%")

    # Per-class accuracy
    print(f"\n   Per-class accuracy:")
    for binary_label in [0, 1]:
        mask = all_binary_labels == binary_label
        if mask.sum() > 0:
            class_acc = (predicted_binary[mask] == all_binary_labels[mask]).mean() * 100
            print(f"      {BINARY_NAMES[binary_label]}: {class_acc:.2f}% ({mask.sum()} images)")

    print("\n8. Creating confusion matrix...")

    cm = confusion_matrix(all_binary_labels, predicted_binary)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No_Tumor', 'Tumor'],
        yticklabels=['No_Tumor', 'Tumor'],
        annot_kws={'size': 16}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Binary Classification: Tumor vs No_Tumor\nAccuracy: {accuracy * 100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(eval_dir, 'confusion_matrix_binary.png')
    plt.savefig(cm_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {cm_path}")

    print("\n9. Creating accuracy chart...")

    class_accuracies = []
    for binary_label in [0, 1]:
        mask = all_binary_labels == binary_label
        if mask.sum() > 0:
            acc = (predicted_binary[mask] == all_binary_labels[mask]).mean() * 100
            class_accuracies.append(acc)

    plt.figure(figsize=(8, 6))
    colors = ['#4CAF50', '#F44336']  # Green for No_Tumor, Red for Tumor
    bars = plt.bar(['No_Tumor', 'Tumor'], class_accuracies, color=colors, edgecolor='black')

    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=14)

    plt.axhline(y=accuracy * 100, color='blue', linestyle='--',
                label=f'Overall: {accuracy * 100:.1f}%', linewidth=2)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Binary Classification Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    acc_path = os.path.join(eval_dir, 'accuracy_binary.png')
    plt.savefig(acc_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {acc_path}")

    print("\n   Creating binary cluster plot (PCA)...")

    from sklearn.decomposition import PCA

    # Combine all features for visualization
    all_features = np.vstack([test_features, val_features])

    # Scale features
    all_scaled = scaler.transform(all_features)

    # Reduce to 2D with PCA
    pca = PCA(n_components=2, random_state=Config.RANDOM_SEED)
    features_2d = pca.fit_transform(all_scaled)

    variance_explained = pca.explained_variance_ratio_.sum() * 100

    # Create figure with 2 subplots: Predicted vs True
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    binary_colors = ['#4CAF50', '#F44336']  # Green=No_Tumor, Red=Tumor

    # Left plot: Predicted binary labels
    ax1 = axes[0]
    for binary_label in [0, 1]:
        mask = predicted_binary == binary_label
        ax1.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=binary_colors[binary_label],
            label=f'{BINARY_NAMES[binary_label]} (n={mask.sum()})',
            alpha=0.5,
            s=30
        )

    ax1.set_xlabel('PCA Component 1', fontsize=12)
    ax1.set_ylabel('PCA Component 2', fontsize=12)
    ax1.set_title('Predicted Binary Labels', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Right plot: True binary labels
    ax2 = axes[1]
    for binary_label in [0, 1]:
        mask = all_binary_labels == binary_label
        ax2.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=binary_colors[binary_label],
            label=f'{BINARY_NAMES[binary_label]} (n={mask.sum()})',
            alpha=0.5,
            s=30
        )

    ax2.set_xlabel('PCA Component 1', fontsize=12)
    ax2.set_ylabel('PCA Component 2', fontsize=12)
    ax2.set_title('True Binary Labels', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Binary Classification: Tumor vs No_Tumor\nPCA Variance Explained: {variance_explained:.1f}% | Accuracy: {accuracy * 100:.1f}%',
        fontsize=14, fontweight='bold')
    plt.tight_layout()

    cluster_plot_path = os.path.join(eval_dir, 'binary_clusters_pca.png')
    plt.savefig(cluster_plot_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {cluster_plot_path}")

    # Also create a single plot showing correct vs incorrect predictions
    plt.figure(figsize=(10, 8))

    correct_mask = predicted_binary == all_binary_labels
    incorrect_mask = ~correct_mask

    plt.scatter(
        features_2d[correct_mask, 0],
        features_2d[correct_mask, 1],
        c='#4CAF50',
        label=f'Correct ({correct_mask.sum()})',
        alpha=0.5,
        s=30
    )
    plt.scatter(
        features_2d[incorrect_mask, 0],
        features_2d[incorrect_mask, 1],
        c='#F44336',
        label=f'Incorrect ({incorrect_mask.sum()})',
        alpha=0.7,
        s=40,
        marker='x'
    )

    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.title(f'Binary Classification: Correct vs Incorrect Predictions\nAccuracy: {accuracy * 100:.1f}%',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    correct_plot_path = os.path.join(eval_dir, 'binary_correct_vs_incorrect.png')
    plt.savefig(correct_plot_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {correct_plot_path}")

    print("\n10. Saving report...")

    report_path = os.path.join(eval_dir, 'binary_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("BINARY CLASSIFICATION REPORT: TUMOR vs NO_TUMOR\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {Config.MODEL_PATH}\n")
        f.write(f"Latent Dimension: {Config.LATENT_DIM}\n\n")

        f.write("DATASET:\n")
        f.write(f"  Test images: {len(test_features)}\n")
        f.write(f"  Validation images: {len(val_features)}\n")
        f.write(f"  Total: {len(all_labels)}\n")
        f.write(f"  Tumor (glioma + meningioma + pituitary): {tumor_count}\n")
        f.write(f"  No_Tumor: {no_tumor_count}\n\n")

        f.write("CLUSTER TO BINARY MAPPING:\n")
        for c, binary_label in cluster_to_binary.items():
            f.write(f"  Cluster {c} -> {BINARY_NAMES[binary_label]}\n")
        f.write("\n")

        f.write("RESULTS:\n")
        f.write(f"  Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"  Precision: {precision * 100:.2f}%\n")
        f.write(f"  Recall:    {recall * 100:.2f}%\n")
        f.write(f"  F1 Score:  {f1 * 100:.2f}%\n\n")

        f.write("PER-CLASS ACCURACY:\n")
        for binary_label in [0, 1]:
            mask = all_binary_labels == binary_label
            if mask.sum() > 0:
                acc = (predicted_binary[mask] == all_binary_labels[mask]).mean() * 100
                f.write(f"  {BINARY_NAMES[binary_label]}: {acc:.2f}% ({mask.sum()} images)\n")
        f.write("\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("              Pred_No_Tumor  Pred_Tumor\n")
        f.write(f"  True_No_Tumor    {cm[0, 0]:5d}        {cm[0, 1]:5d}\n")
        f.write(f"  True_Tumor       {cm[1, 0]:5d}        {cm[1, 1]:5d}\n\n")

        f.write("INTERPRETATION:\n")
        f.write(f"  Random chance (2 classes): 50%\n")
        f.write(f"  Your result: {accuracy * 100:.1f}%\n")
        if accuracy >= 0.7:
            f.write("  -> Good binary separation!\n")
        elif accuracy >= 0.6:
            f.write("  -> Moderate binary separation.\n")
        else:
            f.write("  -> Weak binary separation.\n")

    print(f"   Saved: {report_path}")

    print("\n" + "=" * 70)
    print("BINARY EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\n📊 RESULTS:")
    print(f"   Accuracy:  {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1 Score:  {f1 * 100:.2f}%")

    print(f"\n🗺️ CLUSTER → BINARY MAPPING:")
    for c, binary_label in cluster_to_binary.items():
        print(f"   Cluster {c} = {BINARY_NAMES[binary_label]}")

    print(f"\n📁 Files saved to: {eval_dir}")

    print(f"\n💡 INTERPRETATION:")
    print(f"   Random chance (2 classes): 50%")
    print(f"   Your result: {accuracy * 100:.1f}%")

    if accuracy >= 0.7:
        print(f"   → Good! Model distinguishes tumor presence well.")
    elif accuracy >= 0.6:
        print(f"   → Moderate. Some ability to detect tumors.")
    else:
        print(f"   → Weak. Tumor detection needs improvement.")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cluster_to_binary': cluster_to_binary,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    run_binary_evaluation()