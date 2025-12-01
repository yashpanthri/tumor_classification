import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from datetime import datetime

from config import Config


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

    # Reconstruct scaler
    scaler_params = np.load(scaler_path)
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']

    return cluster_centers, scaler


def assign_to_clusters(features, cluster_centers, scaler):
    # Scale features using training scaler
    features_scaled = scaler.transform(features)

    # Compute distance to each cluster center
    # For each image, find which center is closest
    distances = np.zeros((len(features), len(cluster_centers)))

    for i, center in enumerate(cluster_centers):
        # Euclidean distance from each point to this center
        distances[:, i] = np.sqrt(np.sum((features_scaled - center) ** 2, axis=1))

    # Assign to nearest cluster
    cluster_assignments = np.argmin(distances, axis=1)

    return cluster_assignments


def majority_vote_mapping(cluster_labels, true_labels, n_clusters):
    mapping = {}

    print("\n   Majority vote for each cluster:")

    for cluster_id in range(n_clusters):
        # Get true labels for all images in this cluster
        mask = cluster_labels == cluster_id
        labels_in_cluster = true_labels[mask]

        if len(labels_in_cluster) == 0:
            mapping[cluster_id] = -1
            print(f"      Cluster {cluster_id}: EMPTY")
            continue

        # Count occurrences of each true label
        unique_labels, counts = np.unique(labels_in_cluster, return_counts=True)

        # Find the most common label
        most_common_idx = np.argmax(counts)
        winner_label = unique_labels[most_common_idx]
        winner_count = counts[most_common_idx]
        total = len(labels_in_cluster)

        mapping[cluster_id] = winner_label

        # Show the voting breakdown
        class_name = Config.CLASS_NAMES_4.get(winner_label, f'Class {winner_label}')
        print(f"      Cluster {cluster_id} → {class_name} ({winner_count}/{total} = {winner_count / total * 100:.1f}%)")

    return mapping


def run_evaluation():

    print("\n" + "=" * 70)
    print("EVALUATION - CLUSTER TO TUMOR TYPE MAPPING")
    print("=" * 70)

    # Setup
    eval_dir = os.path.join(Config.RESULTS_DIR, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    print("\n1. Loading cluster model from training...")
    cluster_centers, scaler = load_cluster_model()
    if cluster_centers is None:
        return

    print(f"   Loaded {len(cluster_centers)} cluster centers")
    print(f"   Each center has {cluster_centers.shape[1]} dimensions")

    print("\n2. Loading test set features...")
    test_features, test_labels, test_filenames = load_features('test')
    if test_features is None:
        return
    print(f"   Loaded {len(test_features)} test images with labels")

    print("\n3. Loading validation set features...")
    val_features, val_labels, val_filenames = load_features('validate')
    if val_features is None:
        return
    print(f"   Loaded {len(val_features)} validation images with labels")

    print("\n4. Assigning images to nearest cluster centers...")

    test_clusters = assign_to_clusters(test_features, cluster_centers, scaler)
    val_clusters = assign_to_clusters(val_features, cluster_centers, scaler)

    print(f"   Test set cluster distribution:")
    for c in range(Config.N_CLUSTERS):
        n = (test_clusters == c).sum()
        print(f"      Cluster {c}: {n} images")

    print(f"   Validation set cluster distribution:")
    for c in range(Config.N_CLUSTERS):
        n = (val_clusters == c).sum()
        print(f"      Cluster {c}: {n} images")

    print("\n5. Mapping clusters to tumor types using majority voting...")

    # Combine test + validation for better mapping
    all_clusters = np.concatenate([test_clusters, val_clusters])
    all_true_labels = np.concatenate([test_labels, val_labels])
    all_filenames = np.concatenate([test_filenames, val_filenames])

    mapping = majority_vote_mapping(all_clusters, all_true_labels, Config.N_CLUSTERS)

    print("\n   Final mapping:")
    for cluster_id, label_id in sorted(mapping.items()):
        class_name = Config.CLASS_NAMES_4.get(label_id, 'Unknown')
        print(f"      Cluster {cluster_id} = {class_name}")

    print("\n6. Generating predictions...")

    # Convert cluster IDs to predicted labels
    predicted_labels = np.array([mapping.get(c, -1) for c in all_clusters])

    print("\n7. Calculating accuracy...")

    accuracy = accuracy_score(all_true_labels, predicted_labels)
    print(f"\n   OVERALL ACCURACY: {accuracy * 100:.2f}%")

    # Per-class accuracy
    print("\n   Per-class accuracy:")
    for class_id in range(Config.N_CLUSTERS):
        mask = all_true_labels == class_id
        if mask.sum() > 0:
            class_acc = (predicted_labels[mask] == all_true_labels[mask]).mean() * 100
            class_name = Config.CLASS_NAMES_4[class_id]
            print(f"      {class_name}: {class_acc:.2f}% ({mask.sum()} images)")

    print("\n8. Creating confusion matrix...")

    cm = confusion_matrix(all_true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[Config.CLASS_NAMES_4[i] for i in range(Config.N_CLUSTERS)],
        yticklabels=[Config.CLASS_NAMES_4[i] for i in range(Config.N_CLUSTERS)]
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {cm_path}")

    print("\n9. Creating per-class accuracy chart...")

    class_accuracies = []
    class_names = []
    for class_id in range(Config.N_CLUSTERS):
        mask = all_true_labels == class_id
        if mask.sum() > 0:
            acc = (predicted_labels[mask] == all_true_labels[mask]).mean() * 100
            class_accuracies.append(acc)
            class_names.append(Config.CLASS_NAMES_4[class_id])

    plt.figure(figsize=(10, 6))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
    bars = plt.bar(range(len(class_accuracies)), class_accuracies, color=colors, edgecolor='black')

    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=12)

    plt.axhline(y=accuracy * 100, color='red', linestyle='--', label=f'Overall: {accuracy * 100:.1f}%')
    plt.xticks(range(len(class_names)), class_names, fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Classification Accuracy by Tumor Type', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    acc_path = os.path.join(eval_dir, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {acc_path}")

    print("\n10. Saving detailed results...")

    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': all_true_labels,
        'true_class': [Config.CLASS_NAMES_4[l] for l in all_true_labels],
        'cluster': all_clusters,
        'predicted_label': predicted_labels,
        'predicted_class': [Config.CLASS_NAMES_4[l] for l in predicted_labels],
        'correct': all_true_labels == predicted_labels
    })

    csv_path = os.path.join(eval_dir, 'evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    print("\n11. Saving evaluation report...")

    report_path = os.path.join(eval_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("CLUSTERING EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DATASET:\n")
        f.write(f"  Test images: {len(test_features)}\n")
        f.write(f"  Validation images: {len(val_features)}\n")
        f.write(f"  Total evaluated: {len(all_true_labels)}\n\n")

        f.write("CLUSTER TO TUMOR MAPPING (Majority Vote):\n")
        for cluster_id, label_id in sorted(mapping.items()):
            class_name = Config.CLASS_NAMES_4.get(label_id, 'Unknown')
            f.write(f"  Cluster {cluster_id} -> {class_name}\n")
        f.write("\n")

        f.write("OVERALL RESULTS:\n")
        f.write(f"  Accuracy: {accuracy * 100:.2f}%\n\n")

        f.write("PER-CLASS ACCURACY:\n")
        for class_id in range(Config.N_CLUSTERS):
            mask = all_true_labels == class_id
            if mask.sum() > 0:
                acc = (predicted_labels[mask] == all_true_labels[mask]).mean() * 100
                class_name = Config.CLASS_NAMES_4[class_id]
                f.write(f"  {class_name}: {acc:.2f}% ({mask.sum()} images)\n")
        f.write("\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("  (rows = true, columns = predicted)\n")
        f.write("  " + " ".join([f"{Config.CLASS_NAMES_4[i][:4]:>6}" for i in range(Config.N_CLUSTERS)]) + "\n")
        for i in range(Config.N_CLUSTERS):
            f.write(f"  {Config.CLASS_NAMES_4[i][:4]:4} ")
            f.write(" ".join([f"{cm[i, j]:6d}" for j in range(Config.N_CLUSTERS)]))
            f.write("\n")
        f.write("\n")

        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(
            all_true_labels,
            predicted_labels,
            target_names=[Config.CLASS_NAMES_4[i] for i in range(Config.N_CLUSTERS)],
            zero_division=0
        ))

    print(f"   Saved: {report_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\n📊 RESULTS:")
    print(f"   Overall Accuracy: {accuracy * 100:.2f}%")

    print(f"\n🗺️ CLUSTER MAPPING:")
    for cluster_id, label_id in sorted(mapping.items()):
        class_name = Config.CLASS_NAMES_4.get(label_id, 'Unknown')
        print(f"   Cluster {cluster_id} → {class_name}")

    print(f"\n📁 Files saved to: {eval_dir}")

    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    print(f"   Random guess (4 classes) = 25% accuracy")
    print(f"   Your result: {accuracy * 100:.1f}%")

    if accuracy >= 0.7:
        print(f"   → Excellent for unsupervised learning!")
    elif accuracy >= 0.5:
        print(f"   → Good! The autoencoder learned meaningful features.")
    elif accuracy >= 0.35:
        print(f"   → Above random chance. Some structure learned.")
    else:
        print(f"   → Near random. Features may need improvement.")

    return {
        'accuracy': accuracy,
        'mapping': mapping,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    run_evaluation()