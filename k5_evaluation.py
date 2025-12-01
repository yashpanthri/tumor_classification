import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import os

from config import Config


def run_evaluation_k5():
    K = 5

    print("\n" + "=" * 70)
    print("EVALUATION - K=5 CLUSTERS")
    print("=" * 70)

    # Paths
    k5_features_dir = os.path.join(Config.FEATURES_DIR, 'k5')
    eval_dir = os.path.join(Config.RESULTS_DIR, 'evaluation_k5')
    os.makedirs(eval_dir, exist_ok=True)

    print("\n1. Loading K=5 cluster model...")

    centers_path = os.path.join(k5_features_dir, 'cluster_centers.npy')
    scaler_path = os.path.join(k5_features_dir, 'scaler_params.npz')

    if not os.path.exists(centers_path):
        print(f"   ERROR: Cluster centers not found at {centers_path}")
        print(f"   Run k_means_clustering_k5.py first!")
        return

    cluster_centers = np.load(centers_path)
    scaler_params = np.load(scaler_path)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']

    print(f"   Loaded {K} cluster centers")
    print(f"   Feature dimensions: {cluster_centers.shape[1]}")

    print("\n2. Loading test set features...")
    test_features = np.load(os.path.join(Config.FEATURES_DIR, 'test_features.npy'))
    test_labels = np.load(os.path.join(Config.FEATURES_DIR, 'test_labels.npy'))
    test_filenames = np.load(os.path.join(Config.FEATURES_DIR, 'test_filenames.npy'), allow_pickle=True)
    print(f"   Loaded {len(test_features)} test images")

    print("\n3. Loading validation set features...")
    val_features = np.load(os.path.join(Config.FEATURES_DIR, 'validate_features.npy'))
    val_labels = np.load(os.path.join(Config.FEATURES_DIR, 'validate_labels.npy'))
    val_filenames = np.load(os.path.join(Config.FEATURES_DIR, 'validate_filenames.npy'), allow_pickle=True)
    print(f"   Loaded {len(val_features)} validation images")

    print("\n4. Assigning images to nearest cluster centers...")

    # Scale features
    test_scaled = scaler.transform(test_features)
    val_scaled = scaler.transform(val_features)

    # Assign to nearest cluster
    def assign_clusters(features, centers):
        distances = np.linalg.norm(features[:, np.newaxis] - centers, axis=2)
        return np.argmin(distances, axis=1)

    test_clusters = assign_clusters(test_scaled, cluster_centers)
    val_clusters = assign_clusters(val_scaled, cluster_centers)

    print(f"   Test set cluster distribution:")
    for i in range(K):
        count = (test_clusters == i).sum()
        print(f"      Cluster {i}: {count} images")

    print(f"   Validation set cluster distribution:")
    for i in range(K):
        count = (val_clusters == i).sum()
        print(f"      Cluster {i}: {count} images")

    # Combine test + validation
    all_features = np.vstack([test_scaled, val_scaled])
    all_labels = np.concatenate([test_labels, val_labels])
    all_clusters = np.concatenate([test_clusters, val_clusters])
    all_filenames = np.concatenate([test_filenames, val_filenames])

    print("\n5. Mapping clusters to tumor types using majority voting...")

    cluster_to_class = {}

    print(f"   Majority vote for each cluster:")
    for cluster_id in range(K):
        mask = all_clusters == cluster_id
        labels_in_cluster = all_labels[mask]

        if len(labels_in_cluster) == 0:
            cluster_to_class[cluster_id] = 0
            continue

        unique, counts = np.unique(labels_in_cluster, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        majority_count = counts[np.argmax(counts)]
        total = len(labels_in_cluster)

        cluster_to_class[cluster_id] = majority_class
        class_name = Config.CLASS_NAMES_4.get(majority_class, f"Class {majority_class}")

        print(
            f"      Cluster {cluster_id} → {class_name} ({majority_count}/{total} = {majority_count / total * 100:.1f}%)")

        # Show breakdown
        print(f"         Breakdown: ", end="")
        for cls_id, cnt in zip(unique, counts):
            cls_name = Config.CLASS_NAMES_4.get(cls_id, f"Class {cls_id}")
            print(f"{cls_name}={cnt}, ", end="")
        print()

    print(f"\n   Final mapping:")
    for c, cls_id in cluster_to_class.items():
        cls_name = Config.CLASS_NAMES_4.get(cls_id, f"Class {cls_id}")
        print(f"      Cluster {c} = {cls_name}")

    # Check which classes are represented
    classes_represented = set(cluster_to_class.values())
    all_classes = set(Config.CLASS_NAMES_4.keys())
    missing_classes = all_classes - classes_represented

    if missing_classes:
        print(f"\n   ⚠️  Missing classes: {[Config.CLASS_NAMES_4[c] for c in missing_classes]}")
    else:
        print(f"\n   ✓ All 4 classes represented!")

    print("\n6. Generating predictions...")

    predicted_labels = np.array([cluster_to_class[c] for c in all_clusters])
    predicted_classes = np.array([Config.CLASS_NAMES_4[p] for p in predicted_labels])
    true_classes = np.array([Config.CLASS_NAMES_4[t] for t in all_labels])

    print("\n7. Calculating accuracy...")

    accuracy = accuracy_score(all_labels, predicted_labels)
    print(f"   OVERALL ACCURACY: {accuracy * 100:.2f}%")

    print(f"\n   Per-class accuracy:")
    for class_id in range(4):
        mask = all_labels == class_id
        if mask.sum() > 0:
            class_acc = (predicted_labels[mask] == all_labels[mask]).mean() * 100
            class_name = Config.CLASS_NAMES_4[class_id]
            print(f"      {class_name}: {class_acc:.2f}% ({mask.sum()} images)")

    print("\n8. Creating confusion matrix...")

    class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    cm = confusion_matrix(true_classes, predicted_classes, labels=class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix (K=5)\nAccuracy: {accuracy * 100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(eval_dir, 'confusion_matrix_k5.png')
    plt.savefig(cm_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {cm_path}")

    print("\n9. Creating per-class accuracy chart...")

    class_accuracies = []
    for class_id in range(4):
        mask = all_labels == class_id
        if mask.sum() > 0:
            class_acc = (predicted_labels[mask] == all_labels[mask]).mean() * 100
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)

    plt.figure(figsize=(10, 6))
    colors = Config.COLORS_4CLASS
    bars = plt.bar(class_names, class_accuracies, color=colors, edgecolor='black')

    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=12)

    plt.axhline(y=accuracy * 100, color='blue', linestyle='--',
                label=f'Overall: {accuracy * 100:.1f}%', linewidth=2)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy (K=5)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    acc_path = os.path.join(eval_dir, 'per_class_accuracy_k5.png')
    plt.savefig(acc_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {acc_path}")

    print("\n10. Saving detailed results...")

    csv_path = os.path.join(eval_dir, 'evaluation_results_k5.csv')
    with open(csv_path, 'w') as f:
        f.write('filename,true_label,true_class,cluster,predicted_label,predicted_class,correct\n')
        for fname, true_lbl, cluster, pred_lbl in zip(all_filenames, all_labels, all_clusters, predicted_labels):
            true_cls = Config.CLASS_NAMES_4[true_lbl]
            pred_cls = Config.CLASS_NAMES_4[pred_lbl]
            correct = true_lbl == pred_lbl
            f.write(f'{fname},{true_lbl},{true_cls},{cluster},{pred_lbl},{pred_cls},{correct}\n')
    print(f"   Saved: {csv_path}")

    # Save report
    report_path = os.path.join(eval_dir, 'evaluation_report_k5.txt')
    with open(report_path, 'w') as f:
        f.write("EVALUATION REPORT (K=5 CLUSTERS)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Cluster to Class Mapping:\n")
        for c, cls_id in cluster_to_class.items():
            cls_name = Config.CLASS_NAMES_4.get(cls_id, f"Class {cls_id}")
            f.write(f"   Cluster {c}: {cls_name}\n")
        f.write(f"\nPer-Class Accuracy:\n")
        for class_id, acc in enumerate(class_accuracies):
            class_name = Config.CLASS_NAMES_4[class_id]
            f.write(f"   {class_name}: {acc:.2f}%\n")
        if missing_classes:
            f.write(f"\nMissing classes: {[Config.CLASS_NAMES_4[c] for c in missing_classes]}\n")
        else:
            f.write(f"\nAll 4 classes represented!\n")
    print(f"   Saved: {report_path}")

    print("\n" + "=" * 70)
    print("K=5 EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 RESULTS:")
    print(f"   Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"\n🗺️ CLUSTER MAPPING:")
    for c, cls_id in cluster_to_class.items():
        cls_name = Config.CLASS_NAMES_4.get(cls_id, f"Class {cls_id}")
        print(f"   Cluster {c} → {cls_name}")
    print(f"\n📁 Files saved to: {eval_dir}")


if __name__ == "__main__":
    run_evaluation_k5()