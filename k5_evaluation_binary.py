import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

from config import Config

BINARY_NAMES = {0: 'No_Tumor', 1: 'Tumor'}


def to_binary(labels):
    binary = np.zeros_like(labels)
    binary[labels == 0] = 1  # glioma → Tumor
    binary[labels == 1] = 1  # meningioma → Tumor
    binary[labels == 2] = 1  # pituitary → Tumor
    binary[labels == 3] = 0  # no_tumor → No_Tumor
    return binary


def run_binary_evaluation_k5():

    K = 5

    print("\n" + "=" * 70)
    print("BINARY EVALUATION: TUMOR vs NO_TUMOR (K=5)")
    print("=" * 70)

    # Paths
    k5_features_dir = os.path.join(Config.FEATURES_DIR, 'k5')
    eval_dir = os.path.join(Config.RESULTS_DIR, 'evaluation_binary_k5')
    os.makedirs(eval_dir, exist_ok=True)

    print("\n1. Loading K=5 cluster model...")

    centers_path = os.path.join(k5_features_dir, 'cluster_centers.npy')
    scaler_path = os.path.join(k5_features_dir, 'scaler_params.npz')

    if not os.path.exists(centers_path):
        print(f"   ERROR: Run k_means_clustering_k5.py first!")
        return

    cluster_centers = np.load(centers_path)
    scaler_params = np.load(scaler_path)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']

    print(f"   Loaded {K} cluster centers")

    print("\n2. Loading test and validation features...")

    test_features = np.load(os.path.join(Config.FEATURES_DIR, 'test_features.npy'))
    test_labels = np.load(os.path.join(Config.FEATURES_DIR, 'test_labels.npy'))
    val_features = np.load(os.path.join(Config.FEATURES_DIR, 'validate_features.npy'))
    val_labels = np.load(os.path.join(Config.FEATURES_DIR, 'validate_labels.npy'))

    print(f"   Test: {len(test_features)} images")
    print(f"   Validation: {len(val_features)} images")

    print("\n3. Assigning images to K=5 clusters...")

    test_scaled = scaler.transform(test_features)
    val_scaled = scaler.transform(val_features)

    def assign_clusters(features, centers):
        distances = np.linalg.norm(features[:, np.newaxis] - centers, axis=2)
        return np.argmin(distances, axis=1)

    test_clusters = assign_clusters(test_scaled, cluster_centers)
    val_clusters = assign_clusters(val_scaled, cluster_centers)

    # Combine
    all_features = np.vstack([test_features, val_features])
    all_labels = np.concatenate([test_labels, val_labels])
    all_clusters = np.concatenate([test_clusters, val_clusters])

    print(f"   Total: {len(all_labels)} images")
    for i in range(K):
        count = (all_clusters == i).sum()
        print(f"      Cluster {i}: {count} images")

    print("\n4. Converting labels to binary...")

    all_binary_labels = to_binary(all_labels)
    tumor_count = (all_binary_labels == 1).sum()
    no_tumor_count = (all_binary_labels == 0).sum()

    print(f"   Tumor: {tumor_count}")
    print(f"   No_Tumor: {no_tumor_count}")

    print("\n5. Mapping clusters to binary using 4-class majority vote...")

    cluster_to_binary = {}
    cluster_to_4class = {}

    for cluster_id in range(K):
        mask = all_clusters == cluster_id
        labels_in_cluster = all_labels[mask]

        if len(labels_in_cluster) == 0:
            cluster_to_binary[cluster_id] = 1
            cluster_to_4class[cluster_id] = -1
            continue

        unique, counts = np.unique(labels_in_cluster, return_counts=True)
        majority_4class = unique[np.argmax(counts)]
        majority_count = counts[np.argmax(counts)]
        total = len(labels_in_cluster)

        cluster_to_4class[cluster_id] = majority_4class

        if majority_4class == 3:  # no_tumor
            cluster_to_binary[cluster_id] = 0
            winner_binary = "No_Tumor"
        else:
            cluster_to_binary[cluster_id] = 1
            winner_binary = "Tumor"

        class_name = Config.CLASS_NAMES_4.get(majority_4class, f"Class {majority_4class}")

        binary_labels = to_binary(labels_in_cluster)
        tumor_in = (binary_labels == 1).sum()
        no_tumor_in = (binary_labels == 0).sum()

        print(f"   Cluster {cluster_id}:")
        print(f"      4-class majority: {class_name} ({majority_count}/{total} = {majority_count / total * 100:.1f}%)")
        print(f"      Binary breakdown: Tumor={tumor_in}, No_Tumor={no_tumor_in}")
        print(f"      → Binary label: {winner_binary}")

    print(f"\n   Final binary mapping:")
    for c, binary_label in cluster_to_binary.items():
        class_4 = Config.CLASS_NAMES_4.get(cluster_to_4class[c], "Unknown")
        print(f"      Cluster {c} ({class_4}) = {BINARY_NAMES[binary_label]}")

    tumor_clusters = sum(1 for v in cluster_to_binary.values() if v == 1)
    no_tumor_clusters = sum(1 for v in cluster_to_binary.values() if v == 0)
    print(f"\n   {tumor_clusters} cluster(s) → Tumor, {no_tumor_clusters} cluster(s): No_Tumor")

    print("\n6. Generating predictions...")

    predicted_binary = np.array([cluster_to_binary[c] for c in all_clusters])

    print("\n7. Calculating metrics...")

    accuracy = accuracy_score(all_binary_labels, predicted_binary)
    precision = precision_score(all_binary_labels, predicted_binary, zero_division=0)
    recall = recall_score(all_binary_labels, predicted_binary, zero_division=0)
    f1 = f1_score(all_binary_labels, predicted_binary, zero_division=0)

    print(f"   BINARY CLASSIFICATION RESULTS (K=5):")
    print(f"   =====================================")
    print(f"   Accuracy:  {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1 Score:  {f1 * 100:.2f}%")

    # Per-class
    print(f"\n   Per-class accuracy:")
    for binary_label in [0, 1]:
        mask = all_binary_labels == binary_label
        if mask.sum() > 0:
            class_acc = (predicted_binary[mask] == all_binary_labels[mask]).mean() * 100
            print(f"      {BINARY_NAMES[binary_label]}: {class_acc:.2f}% ({mask.sum()} images)")

    print("\n8. Creating confusion matrix...")

    cm = confusion_matrix(all_binary_labels, predicted_binary)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No_Tumor', 'Tumor'],
                yticklabels=['No_Tumor', 'Tumor'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Binary Confusion Matrix (K=5)\nAccuracy: {accuracy * 100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(eval_dir, 'confusion_matrix_binary_k5.png')
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
    colors = ['#4CAF50', '#F44336']
    bars = plt.bar(['No_Tumor', 'Tumor'], class_accuracies, color=colors, edgecolor='black')

    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=14)

    plt.axhline(y=accuracy * 100, color='blue', linestyle='--',
                label=f'Overall: {accuracy * 100:.1f}%', linewidth=2)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Binary Classification Accuracy (K=5)', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    acc_path = os.path.join(eval_dir, 'accuracy_binary_k5.png')
    plt.savefig(acc_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {acc_path}")

    print("\n10. Saving report...")

    report_path = os.path.join(eval_dir, 'binary_evaluation_report_k5.txt')
    with open(report_path, 'w') as f:
        f.write("BINARY EVALUATION REPORT (K=5 CLUSTERS)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall:    {recall * 100:.2f}%\n")
        f.write(f"F1 Score:  {f1 * 100:.2f}%\n\n")
        f.write("Cluster Mapping:\n")
        for c, binary_label in cluster_to_binary.items():
            class_4 = Config.CLASS_NAMES_4.get(cluster_to_4class[c], "Unknown")
            f.write(f"   Cluster {c} ({class_4}): {BINARY_NAMES[binary_label]}\n")
    print(f"   Saved: {report_path}")

    print("\n" + "=" * 70)
    print("K=5 BINARY EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 RESULTS:")
    print(f"   Accuracy:  {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1 Score:  {f1 * 100:.2f}%")
    print(f"\n📁 Files saved to: {eval_dir}")


if __name__ == "__main__":
    run_binary_evaluation_k5()