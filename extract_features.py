import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from config import Config
from data_loader import BrainTumorDataset
from Autoencoder import ConvAutoencoder


def extract_latent_features(dataset_name="train", folder_path=None, csv_path=None):

    # Use defaults from Config if not specified
    if folder_path is None:
        folder_path = Config.TRAIN_FOLDER

    print("\n" + "=" * 70)
    print(f"FEATURE EXTRACTION: {dataset_name.upper()}")
    print("=" * 70)

    # Create output directory
    print("\n1. Setting up directories...")
    os.makedirs(Config.FEATURES_DIR, exist_ok=True)
    print(f"   ✓ Output: {Config.FEATURES_DIR}")

    # Device setup
    print(f"\n2. Device setup...")
    device = Config.DEVICE
    print(f"   ✓ Using: {device}")

    # Load model
    print(f"\n3. Loading trained autoencoder...")
    if not os.path.exists(Config.MODEL_PATH):
        print(f"   ✗ Model not found: {Config.MODEL_PATH}")
        return None, None, None

    model = ConvAutoencoder(latent_dim=Config.LATENT_DIM)
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"   ✓ Loaded: {Config.MODEL_PATH}")
    print(f"   ✓ Latent dim: {Config.LATENT_DIM}")
    print(f"   ✓ Trained epochs: {checkpoint.get('epoch', 'unknown')}")
    print(
        f"   ✓ Best loss: {checkpoint.get('loss', 'unknown'):.6f}" if isinstance(checkpoint.get('loss'), float) else "")

    # Load dataset
    print(f"\n4. Loading dataset...")

    # Use same transforms as training (no augmentation, no normalization for reconstruction)
    extract_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = BrainTumorDataset(
        folder_path=folder_path,
        csv_path=csv_path,
        transform=extract_transform,
        load_labels=(csv_path is not None)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Keep order consistent
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    print(f"   ✓ Images: {len(dataset)}")
    print(f"   ✓ Batches: {len(dataloader)}")

    # Extract features
    print(f"\n5. Extracting latent features...")

    all_features = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(dataloader):
            images = images.to(device)

            # Extract latent representation
            latent = model.encode(images)

            # Store results
            all_features.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())
            all_filenames.extend([os.path.basename(p) for p in paths])

            # Progress update
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
                processed = min((batch_idx + 1) * Config.BATCH_SIZE, len(dataset))
                print(f"   Processed {processed}/{len(dataset)} images")

    # Combine batches
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    all_filenames = np.array(all_filenames)

    print(f"\n6. Feature extraction complete!")
    print(f"   ✓ Features shape: {all_features.shape}")
    print(f"   ✓ Labels shape: {all_labels.shape}")

    # Save features
    print(f"\n7. Saving to {Config.FEATURES_DIR}...")

    features_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_features.npy')
    labels_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_labels.npy')
    filenames_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_filenames.npy')

    np.save(features_path, all_features)
    np.save(labels_path, all_labels)
    np.save(filenames_path, all_filenames)

    print(f"   ✓ {features_path}")
    print(f"   ✓ {labels_path}")
    print(f"   ✓ {filenames_path}")

    # Feature statistics
    print(f"\n8. Feature statistics:")
    print(f"   Range: [{all_features.min():.4f}, {all_features.max():.4f}]")
    print(f"   Mean:  {all_features.mean():.4f}")
    print(f"   Std:   {all_features.std():.4f}")

    # Visualizations
    print(f"\n9. Creating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature distribution
    axes[0].hist(all_features.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{dataset_name.title()} - Latent Feature Distribution')
    axes[0].grid(True, alpha=0.3)

    # Mean per dimension
    feature_means = np.mean(all_features, axis=0)
    axes[1].plot(feature_means, marker='o', markersize=2, linewidth=0.5)
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Mean Value')
    axes[1].set_title(f'{dataset_name.title()} - Mean per Dimension')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    vis_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_feature_distribution.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {vis_path}")

    # Label distribution (if labels exist)
    valid_labels = all_labels[all_labels >= 0]
    if len(valid_labels) > 0:
        unique, counts = np.unique(valid_labels, return_counts=True)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(unique, counts, edgecolor='black', alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(f'{dataset_name.title()} - Class Distribution')
        plt.xticks(unique, [Config.CLASS_NAMES_4.get(int(u), f'Class {u}') for u in unique], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha='center', fontweight='bold')

        plt.tight_layout()
        class_path = os.path.join(Config.FEATURES_DIR, f'{dataset_name}_class_distribution.png')
        plt.savefig(class_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {class_path}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"   {len(dataset)} images → {Config.LATENT_DIM} features each")
    print(f"   Saved to: {Config.FEATURES_DIR}")

    return all_features, all_labels, all_filenames


def extract_all_datasets():

    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES FROM ALL DATASETS")
    print("=" * 70)

    results = {}

    # Training set (no labels used during training)
    print("\n" + "-" * 70)
    print("TRAINING SET")
    print("-" * 70)
    results['train'] = extract_latent_features(
        dataset_name='train',
        folder_path=Config.TRAIN_FOLDER,
        csv_path=None  # No labels for training data
    )

    # Test set (with labels for evaluation)
    print("\n" + "-" * 70)
    print("TEST SET")
    print("-" * 70)
    results['test'] = extract_latent_features(
        dataset_name='test',
        folder_path=Config.TEST_FOLDER,
        csv_path=Config.TEST_CSV
    )

    # Validation set (with labels for evaluation)
    print("\n" + "-" * 70)
    print("VALIDATION SET")
    print("-" * 70)
    results['validate'] = extract_latent_features(
        dataset_name='validate',
        folder_path=Config.VALIDATE_FOLDER,
        csv_path=Config.VALIDATE_CSV
    )

    # Summary
    print("\n" + "=" * 70)
    print("ALL EXTRACTIONS COMPLETE")
    print("=" * 70)
    print(f"\nFiles saved to: {Config.FEATURES_DIR}")
    print("\nDataset sizes:")
    for name, (features, labels, filenames) in results.items():
        if features is not None:
            print(f"   {name}: {features.shape[0]} images × {features.shape[1]} features")

    print("\nNext step: Run k_means_clustering.py")

    return results


if __name__ == "__main__":
    # Extract from all datasets
    extract_all_datasets()

    # Or extract from just training:
    # extract_latent_features(dataset_name='train', folder_path=Config.TRAIN_FOLDER)