import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from PIL import Image
import os

from config import Config


def demo_single_image(image_path):

    print(f"\n{'=' * 60}")
    print("SVD SINGLE IMAGE DEMO")
    print(f"{'=' * 60}")
    print(f"\nImage: {image_path}")

    # Load image
    img = Image.open(image_path)

    # Convert to grayscale
    if img.mode == 'RGB' or img.mode == 'RGBA':
        img_gray = img.convert('L')
    else:
        img_gray = img

    # Convert to double precision
    X = np.array(img_gray, dtype=np.float64)
    nx, ny = X.shape

    print(f"Image dimensions: {nx} x {ny}")

    # Compute SVD
    print("\nComputing SVD...")
    U, s, Vt = la.svd(X, full_matrices=False)

    print(f"Number of singular values: {len(s)}")
    print(f"Top 5 singular values: {s[:5].round(2)}")

    # Calculate cumulative energy: cumsum(s) / sum(s)
    total_energy = np.sum(s)
    cumulative_energy = np.cumsum(s) / total_energy

    # Find ranks for 95% and 99% energy
    rank_95 = np.argmax(cumulative_energy >= 0.95) + 1
    rank_99 = np.argmax(cumulative_energy >= 0.99) + 1

    print(f"\nEnergy calculation: cumsum(s) / sum(s)")
    print(f"   95% energy: rank = {rank_95} (of {len(s)})")
    print(f"   99% energy: rank = {rank_99} (of {len(s)})")
    print(f"   Compression: {rank_95}/{len(s)} = {rank_95 / len(s) * 100:.1f}% of components for 95%")

    # Reconstruct images
    X_95 = U[:, :rank_95] @ np.diag(s[:rank_95]) @ Vt[:rank_95, :]
    X_99 = U[:, :rank_99] @ np.diag(s[:rank_99]) @ Vt[:rank_99, :]

    # Clip to valid range
    X_95 = np.clip(X_95, 0, 255)
    X_99 = np.clip(X_99, 0, 255)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original color
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Original grayscale
    axes[0, 1].imshow(X, cmap='gray')
    axes[0, 1].set_title('Grayscale Original', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 95% reconstruction
    axes[1, 0].imshow(X_95, cmap='gray')
    axes[1, 0].set_title(f'95% Energy (r={rank_95})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # 99% reconstruction
    axes[1, 1].imshow(X_99, cmap='gray')
    axes[1, 1].set_title(f'99% Energy (r={rank_99})', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle('SVD Image Compression', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Singular values (log scale)
    axes[0].semilogy(s, 'k-', linewidth=1.5)  # Black line
    axes[0].axvline(x=rank_95, color='orange', linestyle='--', label=f'95% (r={rank_95})', linewidth=2)
    axes[0].axvline(x=rank_99, color='green', linestyle='--', label=f'99% (r={rank_99})', linewidth=2)
    # Add filled circles at intersection points
    axes[0].plot(rank_95, s[rank_95 - 1], 'o', color='orange', markersize=10, zorder=5)
    axes[0].plot(rank_99, s[rank_99 - 1], 'o', color='green', markersize=10, zorder=5)
    axes[0].set_xlabel('Index', fontsize=12)
    axes[0].set_ylabel('Singular Value (log scale)', fontsize=12)
    axes[0].set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative energy
    axes[1].plot(cumulative_energy * 100, 'k-', linewidth=1.5)  # Black line
    axes[1].axhline(y=95, color='orange', linestyle='--', label='95%', linewidth=2)
    axes[1].axhline(y=99, color='green', linestyle='--', label='99%', linewidth=2)
    axes[1].axvline(x=rank_95, color='orange', linestyle=':', alpha=0.7)
    axes[1].axvline(x=rank_99, color='green', linestyle=':', alpha=0.7)
    axes[1].plot(rank_95, 95, 'o', color='orange', markersize=10, zorder=5)
    axes[1].plot(rank_99, 99, 'o', color='green', markersize=10, zorder=5)
    # Add rank labels below the circles
    axes[1].text(rank_95, 89, f'r={rank_95}', ha='center', fontsize=10, fontweight='bold', color='orange')
    axes[1].text(rank_99, 93, f'r={rank_99}', ha='center', fontsize=10, fontweight='bold', color='green')
    axes[1].set_xlabel('Number of Components (rank)', fontsize=12)
    axes[1].set_ylabel('Cumulative Energy (%)', fontsize=12)
    axes[1].set_title('Energy: cumsum(s) / sum(s)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.show()

    ranks_to_show = [1, 5, 10, 20, rank_95, rank_99]
    ranks_to_show = sorted(set([r for r in ranks_to_show if r <= len(s)]))

    n_plots = len(ranks_to_show)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, r in enumerate(ranks_to_show):
        X_r = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        X_r = np.clip(X_r, 0, 255)
        energy_at_r = cumulative_energy[r - 1] * 100

        axes[idx].imshow(X_r, cmap='gray')
        axes[idx].set_title(f'Rank {r} ({energy_at_r:.1f}% energy)', fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Progressive SVD Reconstruction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\n{'=' * 60}")
    print("DEMO COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nIf satisfied, run full preprocessing:")
    print(f"   python svd_preprocessing.py")


if __name__ == "__main__":
    # Find a sample image automatically
    sample_image = None

    # Check if TRAIN_FOLDER has images directly (flat) or in subfolders
    train_dir = Config.TRAIN_FOLDER

    # First, look for images directly in the folder
    for f in os.listdir(train_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_image = os.path.join(train_dir, f)
            break

    # If not found, check subdirectories
    if sample_image is None:
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.isdir(class_dir):
                images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_image = os.path.join(class_dir, images[0])
                    break

    if sample_image:
        demo_single_image(sample_image)
    else:
        print("No sample image found!")
        print("You can also run with a specific path:")
        print("   demo_single_image('path/to/your/image.jpg')")