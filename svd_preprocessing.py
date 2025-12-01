import numpy as np
from numpy import linalg as la
from PIL import Image
import os
from datetime import datetime

from config import Config


def load_image_as_grayscale(image_path):
    img = Image.open(image_path)

    if img.mode == 'RGB' or img.mode == 'RGBA':
        img_gray = img.convert('L')
    else:
        img_gray = img

    X = np.array(img_gray, dtype=np.float64)
    return X


def process_single_image(image_path, energy_level):
    X = load_image_as_grayscale(image_path)

    # Compute SVD
    U, s, Vt = la.svd(X, full_matrices=False)

    # Calculate cumulative energy
    total_energy = np.sum(s)
    cumulative_energy = np.cumsum(s) / total_energy

    # Find rank for target energy
    rank = np.argmax(cumulative_energy >= energy_level) + 1

    # Reconstruct
    Xapprox = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
    Xapprox = np.clip(Xapprox, 0, 255)

    return Xapprox, rank


def process_dataset(dataset_name, input_dir, output_dir, energy_level):
    print(f"\n   Processing {dataset_name}...")

    ranks = []
    count = 0

    # Check if flat structure (images directly in folder) or subfolders
    direct_images = [f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
                     and os.path.isfile(os.path.join(input_dir, f))]

    if direct_images:
        # FLAT STRUCTURE
        os.makedirs(output_dir, exist_ok=True)

        total = len(direct_images)
        for i, img_file in enumerate(direct_images):
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"      {i + 1}/{total} images")

            try:
                img_path = os.path.join(input_dir, img_file)
                Xapprox, rank = process_single_image(img_path, energy_level)

                # Save
                img_out = Image.fromarray(Xapprox.astype(np.uint8), mode='L')
                img_out.save(os.path.join(output_dir, img_file))

                ranks.append(rank)
                count += 1
            except Exception as e:
                print(f"      Error: {img_file} - {e}")

    else:
        # SUBFOLDER STRUCTURE
        class_dirs = [d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d))]

        for class_name in class_dirs:
            class_input = os.path.join(input_dir, class_name)
            class_output = os.path.join(output_dir, class_name)
            os.makedirs(class_output, exist_ok=True)

            image_files = [f for f in os.listdir(class_input)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

            total = len(image_files)
            print(f"      {class_name}: {total} images")

            for i, img_file in enumerate(image_files):
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    print(f"         {i + 1}/{total}")

                try:
                    img_path = os.path.join(class_input, img_file)
                    Xapprox, rank = process_single_image(img_path, energy_level)

                    # Save
                    img_out = Image.fromarray(Xapprox.astype(np.uint8), mode='L')
                    img_out.save(os.path.join(class_output, img_file))

                    ranks.append(rank)
                    count += 1
                except Exception as e:
                    print(f"         Error: {img_file} - {e}")

    return count, ranks


def run_svd_preprocessing():

    # Get energy level from config
    energy_level = Config.SVD_ENERGY_LEVEL
    energy_pct = int(energy_level * 100)

    print("\n" + "=" * 60)
    print("SVD IMAGE PREPROCESSING")
    print("=" * 60)
    print(f"\nEnergy level: {energy_pct}%")
    print(f"Formula: cumsum(s) / sum(s)")

    # Output directory
    svd_output_dir = os.path.join(Config.RESULTS_DIR, f'svd_{energy_pct}')
    print(f"Output: {svd_output_dir}")

    all_ranks = []
    total_count = 0

    # Process Training
    train_output = os.path.join(svd_output_dir, 'Training')
    count, ranks = process_dataset('Training', Config.TRAIN_FOLDER, train_output, energy_level)
    total_count += count
    all_ranks.extend(ranks)

    # Process Testing
    test_output = os.path.join(svd_output_dir, 'Testing')
    count, ranks = process_dataset('Testing', Config.TEST_FOLDER, test_output, energy_level)
    total_count += count
    all_ranks.extend(ranks)

    # Process Validation (if exists)
    if os.path.exists(Config.VALIDATE_FOLDER):
        val_output = os.path.join(svd_output_dir, 'Validation')
        count, ranks = process_dataset('Validation', Config.VALIDATE_FOLDER, val_output, energy_level)
        total_count += count
        all_ranks.extend(ranks)

    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nImages processed: {total_count}")
    print(f"Average rank: {np.mean(all_ranks):.1f}")
    print(f"Rank range: {np.min(all_ranks)} - {np.max(all_ranks)}")
    print(f"\nOutput: {svd_output_dir}")

    # Save simple report
    report_path = os.path.join(svd_output_dir, 'svd_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"SVD Preprocessing Report\n")
        f.write(f"========================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Energy level: {energy_pct}%\n")
        f.write(f"Images processed: {total_count}\n")
        f.write(f"Average rank: {np.mean(all_ranks):.1f}\n")
        f.write(f"Rank range: {np.min(all_ranks)} - {np.max(all_ranks)}\n")

    print(f"Report: {report_path}")


if __name__ == "__main__":
    run_svd_preprocessing()