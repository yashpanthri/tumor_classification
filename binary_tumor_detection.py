import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from config import Config
from Autoencoder import ConvAutoencoder


class BinaryTumorDetector:
    def __init__(self, pixel_spacing_mm=0.5):

        self.device = Config.DEVICE
        self.pixel_spacing = pixel_spacing_mm

        # Load autoencoder from config paths
        print("Loading autoencoder model...")
        self.model = ConvAutoencoder(latent_dim=Config.LATENT_DIM)
        checkpoint = torch.load(Config.MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load cluster centers from config paths
        print("Loading cluster centers...")
        cluster_path = os.path.join(Config.FEATURES_DIR, "k5", "cluster_centers.npy")
        self.cluster_centers = np.load(cluster_path)

        # Load scaler from config paths
        print("Loading scaler...")
        scaler_path = os.path.join(Config.FEATURES_DIR, "k5", "scaler_params.npz")
        scaler_data = np.load(scaler_path)
        self.scaler_mean = scaler_data['mean']
        self.scaler_std = scaler_data['scale']  # Fixed: 'scale' not 'std'

        # Transform using config image size
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        print(f"✓ Detector initialized on {self.device}")
        print(f"✓ Model: {Config.MODEL_PATH}")
        print(f"✓ Latent dim: {Config.LATENT_DIM}")
        print(f"✓ Pixel spacing: {pixel_spacing_mm} mm")

    def set_cluster_mapping(self, cluster_to_binary):
        self.cluster_to_binary = cluster_to_binary

    def classify_binary(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            reconstructed, latent = self.model(image_tensor)

            # Get latent features
            features = latent.cpu().numpy().flatten()

            # Standardize
            features_scaled = (features - self.scaler_mean) / self.scaler_std

            # Find nearest cluster
            distances = np.linalg.norm(self.cluster_centers - features_scaled, axis=1)
            cluster_id = np.argmin(distances)

            # Calculate confidence
            min_dist = distances[cluster_id]
            confidence = 1.0 / (1.0 + min_dist)

            # Binary classification
            is_tumor = self.cluster_to_binary[cluster_id] == 'tumor'

            return is_tumor, cluster_id, confidence, reconstructed

    def find_tumor_region(self, image_tensor, reconstructed):
        # Convert tensors to numpy
        original_np = image_tensor.numpy().transpose(1, 2, 0)
        reconstructed_np = reconstructed.cpu().numpy().squeeze().transpose(1, 2, 0)

        # Calculate reconstruction error
        error = np.abs(original_np - reconstructed_np)
        error_gray = np.mean(error, axis=2)

        # Normalize error to 0-255
        error_norm = ((error_gray - error_gray.min()) /
                      (error_gray.max() - error_gray.min() + 1e-8) * 255).astype(np.uint8)

        # Convert original to grayscale
        gray = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Threshold masks
        _, intensity_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, error_thresh = cv2.threshold(error_norm, 100, 255, cv2.THRESH_BINARY)

        # Combine masks
        combined = cv2.bitwise_and(intensity_thresh, error_thresh)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            contours, _ = cv2.findContours(intensity_thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None, error_norm

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Filter small contours
        if cv2.contourArea(largest_contour) < 500:
            return None, None, None, error_norm

        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        area_pixels = cv2.contourArea(largest_contour)

        return (x, y, w, h), largest_contour, area_pixels, error_norm

    def calculate_size(self, bbox, area_pixels):
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

    def analyze_image(self, image_path):
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        # Binary classification
        is_tumor, cluster_id, confidence, reconstructed = self.classify_binary(image_tensor)

        # Find tumor region (only if tumor detected)
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

    def visualize_result(self, result, save_path=None):
        image_np = result['image_tensor'].numpy().transpose(1, 2, 0)

        if result['is_tumor']:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # 1. Original with bounding box
            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title(f"TUMOR DETECTED\nConfidence: {result['confidence']:.1%}",
                                 fontsize=14, fontweight='bold', color='red')

            if result['bbox']:
                x, y, w, h = result['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                         edgecolor='red', facecolor='none')
                axes[0, 0].add_patch(rect)

                if result['size']:
                    size = result['size']
                    label = f"{size['width_cm']:.1f} x {size['height_cm']:.1f} cm"
                    axes[0, 0].text(x, y - 15, label, color='white', fontsize=11,
                                    fontweight='bold',
                                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
            axes[0, 0].axis('off')

            # 2. Error heatmap
            if result['error_heatmap'] is not None:
                im = axes[0, 1].imshow(result['error_heatmap'], cmap='hot')
                axes[0, 1].set_title("Reconstruction Error\n(Bright = Anomaly)", fontsize=12)
                plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
            else:
                axes[0, 1].imshow(image_np)
                axes[0, 1].set_title("Original Image", fontsize=12)
            axes[0, 1].axis('off')

            # 3. Reconstruction
            reconstructed_np = result['reconstructed'].cpu().numpy().squeeze().transpose(1, 2, 0)
            axes[1, 0].imshow(np.clip(reconstructed_np, 0, 1))
            axes[1, 0].set_title("Autoencoder Reconstruction", fontsize=12)
            axes[1, 0].axis('off')

            # 4. Overlay with contour
            overlay = (image_np * 255).astype(np.uint8).copy()
            if result['contour'] is not None:
                cv2.drawContours(overlay, [result['contour']], -1, (255, 0, 0), 3)
                if result['bbox']:
                    x, y, w, h = result['bbox']
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title("Detected Region", fontsize=12)
            axes[1, 1].axis('off')

            summary = f"File: {os.path.basename(result['image_path'])}\n"
            if result['size']:
                summary += f"Estimated Size: {result['size']['diameter_cm']:.2f} cm diameter | "
                summary += f"Area: {result['size']['area_cm2']:.2f} cm2"

        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image_np)
            ax.set_title(f"NO TUMOR DETECTED\nConfidence: {result['confidence']:.1%}",
                         fontsize=16, fontweight='bold', color='green')
            ax.axis('off')
            summary = f"File: {os.path.basename(result['image_path'])}"

        fig.suptitle(summary, fontsize=11, y=0.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=Config.FIG_DPI, bbox_inches='tight', facecolor='white')
            print(f"   Saved: {save_path}")

        plt.close()
        return fig


def run_detection(input_folder, output_folder, cluster_to_binary):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"BINARY TUMOR DETECTION")
    print(f"{'=' * 70}")
    print(f"Model: {Config.MODEL_PATH}")
    print(f"Results Dir: {Config.RESULTS_DIR}")

    # Initialize detector (uses config.py internally)
    detector = BinaryTumorDetector(pixel_spacing_mm=0.5)
    detector.set_cluster_mapping(cluster_to_binary)

    # Find images
    image_files = [f for f in os.listdir(input_folder)
                   if os.path.splitext(f)[1].lower() in Config.IMAGE_EXTENSIONS]

    print(f"\nFound {len(image_files)} images to analyze\n")

    # Analyze each image
    results = []
    tumor_count = 0

    for i, filename in enumerate(image_files):
        print(f"[{i + 1}/{len(image_files)}] {filename}")

        image_path = os.path.join(input_folder, filename)
        result = detector.analyze_image(image_path)
        results.append(result)

        if result['is_tumor']:
            tumor_count += 1
            print(f"   TUMOR DETECTED (confidence: {result['confidence']:.1%})")
            if result['size']:
                print(f"   Size: {result['size']['diameter_cm']:.2f} cm diameter")
        else:
            print(f"   No tumor (confidence: {result['confidence']:.1%})")

        # Save visualization
        save_path = os.path.join(output_folder, f"result_{os.path.splitext(filename)[0]}.png")
        detector.visualize_result(result, save_path)

    # Generate report
    report_path = os.path.join(output_folder, "detection_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BINARY TUMOR DETECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model Path: {Config.MODEL_PATH}\n")
        f.write(f"Results Dir: {Config.RESULTS_DIR}\n")
        f.write(f"Images Analyzed: {len(results)}\n")
        f.write(f"Tumors Detected: {tumor_count}\n")
        f.write(f"No Tumor: {len(results) - tumor_count}\n")
        f.write(f"Pixel Spacing: 0.5 mm (assumed)\n\n")

        f.write("-" * 70 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 70 + "\n\n")

        for r in results:
            f.write(f"File: {os.path.basename(r['image_path'])}\n")
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
    print(f"\n   Results saved to: {output_folder}")
    print(f"   Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    # ===== INPUT/OUTPUT FOLDERS =====
    INPUT_FOLDER = os.path.join(Config.BASE_DIR, "new_images")
    OUTPUT_FOLDER = os.path.join(Config.RESULTS_DIR, "detection_results")

    # ===== CLUSTER MAPPINGS =====
    # H9 model mapping (from H9 evaluation results)
    CLUSTER_TO_BINARY_H9 = {
        0: 'tumor',  # pituitary
        1: 'no_tumor',  # no_tumor
        2: 'tumor',  # pituitary
        3: 'tumor',  # meningioma
        4: 'tumor'  # glioma
    }

    # SVD95 model mapping (from SVD95 evaluation results)
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
        # Select cluster mapping based on config
        if 'SVD95' in Config.RESULTS_DIR:
            cluster_mapping = CLUSTER_TO_BINARY_SVD95
            print("Using SVD95 cluster mapping")
        else:
            cluster_mapping = CLUSTER_TO_BINARY_H9
            print("Using H9 cluster mapping")

        # Run detection
        results = run_detection(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            cluster_to_binary=cluster_mapping
        )