"""
Binary Tumor Detection with SVD Preprocessing
Uses the same SVD method as svd_demo.py and svd_preprocessing.py
1. Converts to grayscale
2. Applies SVD with energy-based rank selection
3. Runs binary tumor detection
"""

import os
import torch
import numpy as np
from numpy import linalg as la
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from config import Config
from Autoencoder import ConvAutoencoder


class SVDPreprocessor:
    """Applies SVD preprocessing using same method as svd_demo.py"""

    def __init__(self, energy_threshold=0.95):
        """
        Args:
            energy_threshold: Energy level to retain (0.95 = 95%)
        """
        self.energy_threshold = energy_threshold
        print(f"✓ SVD Preprocessor initialized (retaining {energy_threshold * 100:.0f}% energy)")

    def preprocess_image(self, image_path):
        """
        Apply SVD preprocessing using cumsum(s)/sum(s) energy calculation
        Matches the method in svd_demo.py

        Args:
            image_path: Path to input image

        Returns:
            preprocessed_image: PIL Image after SVD reconstruction (RGB)
            info: Dict with preprocessing details
        """
        # Load image
        img = Image.open(image_path)

        # Store original for visualization
        original_rgb = img.convert('RGB')

        # Convert to grayscale (same as demo)
        if img.mode == 'RGB' or img.mode == 'RGBA':
            img_gray = img.convert('L')
        else:
            img_gray = img

        # Convert to double precision (same as demo)
        X = np.array(img_gray, dtype=np.float64)
        nx, ny = X.shape

        # Compute SVD (same as demo)
        U, s, Vt = la.svd(X, full_matrices=False)

        # Calculate cumulative energy: cumsum(s) / sum(s) (same as demo)
        total_energy = np.sum(s)
        cumulative_energy = np.cumsum(s) / total_energy

        # Find rank for desired energy level (same as demo)
        rank = np.argmax(cumulative_energy >= self.energy_threshold) + 1

        # Reconstruct image: U[:, :rank] @ diag(s[:rank]) @ Vt[:rank, :] (same as demo)
        X_reconstructed = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

        # Clip to valid range (same as demo)
        X_reconstructed = np.clip(X_reconstructed, 0, 255).astype(np.uint8)

        # Convert grayscale back to RGB (3-channel) for autoencoder
        # Stack grayscale into 3 channels
        reconstructed_rgb = np.stack([X_reconstructed, X_reconstructed, X_reconstructed], axis=2)

        # Convert to PIL Image
        preprocessed_image = Image.fromarray(reconstructed_rgb, mode='RGB')

        info = {
            'original_path': image_path,
            'energy_retained': self.energy_threshold,
            'rank_used': rank,
            'total_singular_values': len(s),
            'compression_ratio': rank / len(s),
            'image_size': (nx, ny),
            'actual_energy': cumulative_energy[rank - 1]
        }

        return preprocessed_image, info, original_rgb

    def preprocess_image_rgb(self, image_path):
        """
        Apply SVD preprocessing to each RGB channel separately
        Alternative method if model was trained on RGB SVD images

        Args:
            image_path: Path to input image

        Returns:
            preprocessed_image: PIL Image after SVD reconstruction (RGB)
            info: Dict with preprocessing details
        """
        # Load image as RGB
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float64)

        # Process each channel separately
        reconstructed_channels = []
        ranks_used = []

        for c in range(3):  # R, G, B
            X = img_array[:, :, c]

            # Compute SVD
            U, s, Vt = la.svd(X, full_matrices=False)

            # Calculate cumulative energy: cumsum(s) / sum(s)
            total_energy = np.sum(s)
            cumulative_energy = np.cumsum(s) / total_energy

            # Find rank for desired energy level
            rank = np.argmax(cumulative_energy >= self.energy_threshold) + 1
            ranks_used.append(rank)

            # Reconstruct
            X_reconstructed = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
            X_reconstructed = np.clip(X_reconstructed, 0, 255)
            reconstructed_channels.append(X_reconstructed)

        # Combine channels
        reconstructed_rgb = np.stack(reconstructed_channels, axis=2).astype(np.uint8)
        preprocessed_image = Image.fromarray(reconstructed_rgb, mode='RGB')

        info = {
            'original_path': image_path,
            'energy_retained': self.energy_threshold,
            'ranks_per_channel': ranks_used,
            'avg_rank': np.mean(ranks_used),
            'image_size': img_array.shape[:2]
        }

        return preprocessed_image, info, img


class BinaryTumorDetector:
    """Binary tumor detection using trained autoencoder"""

    def __init__(self, pixel_spacing_mm=0.5):
        """Initialize detector using paths from config.py"""
        self.device = Config.DEVICE
        self.pixel_spacing = pixel_spacing_mm

        # Load autoencoder
        print("Loading autoencoder model...")
        self.model = ConvAutoencoder(latent_dim=Config.LATENT_DIM)
        checkpoint = torch.load(Config.MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load cluster centers
        print("Loading cluster centers...")
        cluster_path = os.path.join(Config.FEATURES_DIR, "k5", "cluster_centers.npy")
        self.cluster_centers = np.load(cluster_path)

        # Load scaler
        print("Loading scaler...")
        scaler_path = os.path.join(Config.FEATURES_DIR, "k5", "scaler_params.npz")
        scaler_data = np.load(scaler_path)
        self.scaler_mean = scaler_data['mean']
        self.scaler_std = scaler_data['scale']

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        print(f"✓ Detector initialized on {self.device}")
        print(f"✓ Model: {Config.MODEL_PATH}")
        print(f"✓ Latent dim: {Config.LATENT_DIM}")

    def set_cluster_mapping(self, cluster_to_binary):
        """Set the cluster to binary mapping"""
        self.cluster_to_binary = cluster_to_binary

    def classify_binary(self, image_tensor):
        """Binary classification: Tumor or No Tumor"""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            reconstructed, latent = self.model(image_tensor)

            features = latent.cpu().numpy().flatten()
            features_scaled = (features - self.scaler_mean) / self.scaler_std

            distances = np.linalg.norm(self.cluster_centers - features_scaled, axis=1)
            cluster_id = np.argmin(distances)

            min_dist = distances[cluster_id]
            confidence = 1.0 / (1.0 + min_dist)

            is_tumor = self.cluster_to_binary[cluster_id] == 'tumor'

            return is_tumor, cluster_id, confidence, reconstructed

    def find_tumor_region(self, image_tensor, reconstructed):
        """Find tumor region using reconstruction error + intensity"""
        original_np = image_tensor.numpy().transpose(1, 2, 0)
        reconstructed_np = reconstructed.cpu().numpy().squeeze().transpose(1, 2, 0)

        error = np.abs(original_np - reconstructed_np)
        error_gray = np.mean(error, axis=2)

        error_norm = ((error_gray - error_gray.min()) /
                      (error_gray.max() - error_gray.min() + 1e-8) * 255).astype(np.uint8)

        gray = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        _, intensity_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, error_thresh = cv2.threshold(error_norm, 100, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_and(intensity_thresh, error_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            contours, _ = cv2.findContours(intensity_thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None, error_norm

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 500:
            return None, None, None, error_norm

        x, y, w, h = cv2.boundingRect(largest_contour)
        area_pixels = cv2.contourArea(largest_contour)

        return (x, y, w, h), largest_contour, area_pixels, error_norm

    def calculate_size(self, bbox, area_pixels):
        """Convert pixel measurements to metric units"""
        if bbox is None:
            return None

        x, y, w, h = bbox

        width_mm = w * self.pixel_spacing
        height_mm = h * self.pixel_spacing
        area_mm2 = area_pixels * (self.pixel_spacing ** 2)
        diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)

        return {
            'width_cm': width_mm / 10,
            'height_cm': height_mm / 10,
            'area_cm2': area_mm2 / 100,
            'diameter_cm': diameter_mm / 10,
            'bbox_pixels': bbox
        }

    def analyze_image(self, image):
        """Analyze a single image (PIL Image or path)"""
        if isinstance(image, str):
            image_pil = Image.open(image).convert('RGB')
            image_path = image
        else:
            image_pil = image
            image_path = "preprocessed_image"

        image_tensor = self.transform(image_pil)

        is_tumor, cluster_id, confidence, reconstructed = self.classify_binary(image_tensor)

        if is_tumor:
            bbox, contour, area_pixels, error_heatmap = self.find_tumor_region(
                image_tensor, reconstructed)
            size_info = self.calculate_size(bbox, area_pixels)
        else:
            bbox, contour, error_heatmap = None, None, None
            size_info = None

        return {
            'image_path': image_path,
            'is_tumor': is_tumor,
            'classification': 'TUMOR' if is_tumor else 'NO TUMOR',
            'cluster_id': cluster_id,
            'confidence': confidence,
            'bbox': bbox,
            'contour': contour,
            'size': size_info,
            'image_tensor': image_tensor,
            'reconstructed': reconstructed,
            'error_heatmap': error_heatmap
        }

    def visualize_result(self, result, original_image=None, svd_image=None,
                         svd_info=None, save_path=None):
        """Create visualization with original, SVD preprocessed, and detection results"""
        image_np = result['image_tensor'].numpy().transpose(1, 2, 0)

        if result['is_tumor']:
            fig, axes = plt.subplots(2, 3, figsize=(16, 11))

            # Row 1: Pipeline
            if original_image is not None:
                axes[0, 0].imshow(original_image)
                axes[0, 0].set_title("1. Original Image", fontsize=12)
            else:
                axes[0, 0].imshow(image_np)
                axes[0, 0].set_title("Input Image", fontsize=12)
            axes[0, 0].axis('off')

            if svd_image is not None:
                axes[0, 1].imshow(svd_image)
                if svd_info:
                    title = f"2. SVD Preprocessed\n(rank={svd_info['rank_used']}, {svd_info['energy_retained'] * 100:.0f}% energy)"
                else:
                    title = "2. SVD Preprocessed"
                axes[0, 1].set_title(title, fontsize=12)
            else:
                axes[0, 1].imshow(image_np)
                axes[0, 1].set_title("SVD Preprocessed", fontsize=12)
            axes[0, 1].axis('off')

            axes[0, 2].imshow(image_np)
            axes[0, 2].set_title(f"3. TUMOR DETECTED\nConfidence: {result['confidence']:.1%}",
                                 fontsize=12, fontweight='bold', color='red')

            if result['bbox']:
                x, y, w, h = result['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                         edgecolor='red', facecolor='none')
                axes[0, 2].add_patch(rect)

                if result['size']:
                    size = result['size']
                    label = f"{size['width_cm']:.1f} x {size['height_cm']:.1f} cm"
                    axes[0, 2].text(x, y - 15, label, color='white', fontsize=10,
                                    fontweight='bold',
                                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
            axes[0, 2].axis('off')

            # Row 2: Analysis
            if result['error_heatmap'] is not None:
                im = axes[1, 0].imshow(result['error_heatmap'], cmap='hot')
                axes[1, 0].set_title("Reconstruction Error\n(Bright = Anomaly)", fontsize=11)
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
            axes[1, 0].axis('off')

            reconstructed_np = result['reconstructed'].cpu().numpy().squeeze().transpose(1, 2, 0)
            axes[1, 1].imshow(np.clip(reconstructed_np, 0, 1))
            axes[1, 1].set_title("Autoencoder Reconstruction", fontsize=11)
            axes[1, 1].axis('off')

            overlay = (image_np * 255).astype(np.uint8).copy()
            if result['contour'] is not None:
                cv2.drawContours(overlay, [result['contour']], -1, (255, 0, 0), 3)
                if result['bbox']:
                    x, y, w, h = result['bbox']
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title("Detected Region Overlay", fontsize=11)
            axes[1, 2].axis('off')

            summary = f"File: {os.path.basename(result['image_path'])}"
            if result['size']:
                summary += f" | Size: {result['size']['diameter_cm']:.2f} cm diameter"
                summary += f" | Area: {result['size']['area_cm2']:.2f} cm2"

        else:
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            if original_image is not None:
                axes[0].imshow(original_image)
                axes[0].set_title("1. Original Image", fontsize=12)
            else:
                axes[0].imshow(image_np)
                axes[0].set_title("Input Image", fontsize=12)
            axes[0].axis('off')

            if svd_image is not None:
                axes[1].imshow(svd_image)
                if svd_info:
                    title = f"2. SVD Preprocessed\n(rank={svd_info['rank_used']}, {svd_info['energy_retained'] * 100:.0f}% energy)"
                else:
                    title = "2. SVD Preprocessed"
                axes[1].set_title(title, fontsize=12)
            else:
                axes[1].imshow(image_np)
            axes[1].axis('off')

            axes[2].imshow(image_np)
            axes[2].set_title(f"3. NO TUMOR DETECTED\nConfidence: {result['confidence']:.1%}",
                              fontsize=12, fontweight='bold', color='green')
            axes[2].axis('off')

            summary = f"File: {os.path.basename(result['image_path'])}"

        fig.suptitle(summary, fontsize=12, y=0.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=Config.FIG_DPI, bbox_inches='tight', facecolor='white')
            print(f"   Saved: {save_path}")

        plt.close()
        return fig


def run_svd_detection(input_folder, output_folder, cluster_to_binary, energy_threshold=0.95):
    """
    Run SVD preprocessing + binary tumor detection
    Uses same SVD method as svd_demo.py: cumsum(s)/sum(s) energy calculation

    Args:
        input_folder: Folder with raw images to analyze
        output_folder: Folder to save results
        cluster_to_binary: Dict mapping cluster_id → 'tumor' or 'no_tumor'
        energy_threshold: Energy level to retain (0.95 = 95%)
    """
    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    svd_folder = os.path.join(output_folder, "svd_preprocessed")
    os.makedirs(svd_folder, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"BINARY TUMOR DETECTION WITH SVD PREPROCESSING")
    print(f"{'=' * 70}")
    print(f"SVD Energy Threshold: {energy_threshold * 100:.0f}%")
    print(f"Energy Calculation: cumsum(s) / sum(s)")
    print(f"Model: {Config.MODEL_PATH}")

    # Initialize preprocessor and detector
    preprocessor = SVDPreprocessor(energy_threshold=energy_threshold)
    detector = BinaryTumorDetector(pixel_spacing_mm=0.5)
    detector.set_cluster_mapping(cluster_to_binary)

    # Find images
    image_files = [f for f in os.listdir(input_folder)
                   if os.path.splitext(f)[1].lower() in Config.IMAGE_EXTENSIONS]

    print(f"\nFound {len(image_files)} images to analyze\n")

    # Process each image
    results = []
    tumor_count = 0

    for i, filename in enumerate(image_files):
        print(f"[{i + 1}/{len(image_files)}] {filename}")

        image_path = os.path.join(input_folder, filename)

        # Step 1: SVD preprocessing (same method as svd_demo.py)
        print(f"   Applying SVD (cumsum(s)/sum(s) >= {energy_threshold * 100:.0f}%)...")
        svd_image, svd_info, original_image = preprocessor.preprocess_image(image_path)
        print(f"   Rank used: {svd_info['rank_used']} of {svd_info['total_singular_values']} "
              f"({svd_info['compression_ratio'] * 100:.1f}% of components)")

        # Save SVD preprocessed image
        svd_save_path = os.path.join(svd_folder, f"svd_{filename}")
        svd_image.save(svd_save_path)

        # Step 2: Run detection on SVD preprocessed image
        result = detector.analyze_image(svd_image)
        result['image_path'] = image_path
        result['svd_info'] = svd_info
        results.append(result)

        # Print results
        if result['is_tumor']:
            tumor_count += 1
            print(f"   TUMOR DETECTED (confidence: {result['confidence']:.1%})")
            if result['size']:
                print(f"   Size: {result['size']['diameter_cm']:.2f} cm diameter")
        else:
            print(f"   No tumor (confidence: {result['confidence']:.1%})")

        # Save visualization
        vis_save_path = os.path.join(output_folder, f"result_{os.path.splitext(filename)[0]}.png")
        detector.visualize_result(result, original_image=original_image,
                                  svd_image=svd_image, svd_info=svd_info,
                                  save_path=vis_save_path)

    # Generate report
    report_path = os.path.join(output_folder, "detection_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BINARY TUMOR DETECTION REPORT (SVD PREPROCESSED)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model Path: {Config.MODEL_PATH}\n")
        f.write(f"SVD Energy Threshold: {energy_threshold * 100:.0f}%\n")
        f.write(f"SVD Method: cumsum(s) / sum(s)\n")
        f.write(f"Images Analyzed: {len(results)}\n")
        f.write(f"Tumors Detected: {tumor_count}\n")
        f.write(f"No Tumor: {len(results) - tumor_count}\n")
        f.write(f"Pixel Spacing: 0.5 mm (assumed)\n\n")

        f.write("-" * 70 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 70 + "\n\n")

        for r in results:
            f.write(f"File: {os.path.basename(r['image_path'])}\n")
            f.write(f"   SVD Rank: {r['svd_info']['rank_used']} of {r['svd_info']['total_singular_values']}\n")
            f.write(f"   Compression: {r['svd_info']['compression_ratio'] * 100:.1f}% of components\n")
            f.write(f"   Result: {r['classification']}\n")
            f.write(f"   Confidence: {r['confidence']:.1%}\n")
            if r['is_tumor'] and r['size']:
                f.write(f"   Bounding Box: {r['bbox']}\n")
                f.write(f"   Width: {r['size']['width_cm']:.2f} cm\n")
                f.write(f"   Height: {r['size']['height_cm']:.2f} cm\n")
                f.write(f"   Diameter: {r['size']['diameter_cm']:.2f} cm\n")
                f.write(f"   Area: {r['size']['area_cm2']:.2f} cm2\n")
            f.write("\n")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"   Total Images: {len(results)}")
    print(f"   Tumors Detected: {tumor_count}")
    print(f"   No Tumor: {len(results) - tumor_count}")
    print(f"\n   SVD Preprocessed images: {svd_folder}")
    print(f"   Results saved to: {output_folder}")
    print(f"   Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    # ===== FOLDERS =====
    INPUT_FOLDER = os.path.join(Config.BASE_DIR, "new_images")
    OUTPUT_FOLDER = os.path.join(Config.RESULTS_DIR, "detection_results")

    # ===== SVD ENERGY THRESHOLD =====
    # Same as your preprocessing: 95% energy retained
    SVD_ENERGY = 0.95

    # ===== CLUSTER MAPPING FOR SVD95 =====
    # From your SVD95 evaluation results
    CLUSTER_TO_BINARY_SVD95 = {
        0: 'no_tumor',  # no_tumor
        1: 'tumor',  # pituitary
        2: 'tumor',  # meningioma
        3: 'tumor',  # pituitary
        4: 'tumor'  # glioma
    }

    # Create input folder if needed
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    # Check for images
    valid_images = [f for f in os.listdir(INPUT_FOLDER)
                    if os.path.splitext(f)[1].lower() in Config.IMAGE_EXTENSIONS]

    if len(valid_images) == 0:
        print(f"\nNo images found in: {INPUT_FOLDER}")
        print(f"Please add images to analyze and run again.")
        print(f"Supported formats: {Config.IMAGE_EXTENSIONS}")
    else:
        # Run detection with SVD preprocessing
        results = run_svd_detection(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            cluster_to_binary=CLUSTER_TO_BINARY_SVD95,
            energy_threshold=SVD_ENERGY
        )