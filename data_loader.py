import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config


def get_transforms(training=True):
    img_size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    if training:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class BrainTumorDataset(Dataset):

    def __init__(self, folder_path, transform=None, load_labels=False, csv_path=None):
        self.folder_path = folder_path
        self.transform = transform
        self.load_labels = load_labels

        # Get all image files from folder
        self.image_files = []
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(filename)

        # Sort for consistency
        self.image_files.sort()

        print(f"Found {len(self.image_files)} images in {folder_path}")

        # Load labels from CSV if requested
        self.labels = {}  # Dictionary: filename -> label_index
        self.label_to_idx = Config.CLASS_TO_IDX_4  # {glioma: 0, meningioma: 1, pituitary: 2, no_tumor: 3}

        if load_labels and csv_path:
            self._load_labels_from_csv(csv_path)
        else:
            # No labels - use -1 for all (unsupervised training)
            for filename in self.image_files:
                self.labels[filename] = -1

    def _load_labels_from_csv(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}")
            # Use -1 for missing labels
            for filename in self.image_files:
                self.labels[filename] = -1
            return

        # Read CSV and create filename -> label mapping
        csv_labels = {}
        with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '').strip()
                label_str = row.get('label', '').strip().lower()

                # Map label string to index
                if label_str in self.label_to_idx:
                    csv_labels[filename] = self.label_to_idx[label_str]
                else:
                    print(f"Warning: Unknown label '{label_str}' for {filename}")
                    csv_labels[filename] = -1

        print(f"Loaded {len(csv_labels)} labels from CSV")

        # Match each image file to its label from CSV
        matched = 0
        for filename in self.image_files:
            if filename in csv_labels:
                self.labels[filename] = csv_labels[filename]
                matched += 1
            else:
                print(f"Warning: No label found for {filename}")
                self.labels[filename] = -1

        print(f"Matched {matched}/{len(self.image_files)} labels from CSV")

        # Show label distribution
        from collections import Counter
        label_counts = Counter([label for label in self.labels.values() if label != -1])
        print(f"Label distribution: {dict(label_counts)}")

        # Show class names
        idx_to_class = {v: k for k, v in self.label_to_idx.items()}
        print(f"Class mapping: ", end="")
        for idx in sorted(label_counts.keys()):
            class_name = idx_to_class.get(idx, 'unknown')
            count = label_counts[idx]
            print(f"{idx}={class_name}({count}), ", end="")
        print()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get filename
        filename = self.image_files[idx]
        img_path = os.path.join(self.folder_path, filename)

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label for this specific filename
        label = self.labels.get(filename, -1)

        return image, label, img_path


def test_data_loader():
    print("=" * 70)
    print("TESTING DATA LOADER - LABEL MATCHING")
    print("=" * 70)

    idx_to_class = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}

    print("\n" + "=" * 70)
    print("TESTING TEST DATASET")
    print("=" * 70)

    test_dataset = BrainTumorDataset(
        folder_path=Config.TEST_FOLDER,
        transform=get_transforms(training=False),
        load_labels=True,
        csv_path=Config.TEST_CSV
    )

    # Check first 10 images
    print(f"\nFirst 10 images and their labels:")
    for i in range(min(10, len(test_dataset))):
        image, label, path = test_dataset[i]
        filename = os.path.basename(path)
        class_name = idx_to_class.get(label, 'unknown')
        print(f"  {i + 1}. {filename}: label={label} ({class_name})")

    # Load CSV to verify
    print(f"\nVerifying against TEST CSV:")
    import csv
    test_csv_data = {}
    with open(Config.TEST_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '').strip()
            label_str = row.get('label', '').strip().lower()
            test_csv_data[filename] = label_str

    # Check if first 10 match
    print(f"CSV labels for first 10 files:")
    test_matches = 0
    for i in range(min(10, len(test_dataset))):
        image, label, path = test_dataset[i]
        filename = os.path.basename(path)
        csv_label = test_csv_data.get(filename, 'NOT FOUND')
        dataset_label = idx_to_class.get(label, 'unknown')
        match = "✓" if csv_label == dataset_label else "✗"
        if csv_label == dataset_label:
            test_matches += 1
        print(f"  {match} {filename}: CSV={csv_label}, Dataset={dataset_label}")

    print(f"\nTEST Dataset: {test_matches}/10 matches ✓")

    print("\n" + "=" * 70)
    print("TESTING VALIDATE DATASET")
    print("=" * 70)

    validate_dataset = BrainTumorDataset(
        folder_path=Config.VALIDATE_FOLDER,
        transform=get_transforms(training=False),
        load_labels=True,
        csv_path=Config.VALIDATE_CSV
    )

    # Check first 10 images
    print(f"\nFirst 10 images and their labels:")
    for i in range(min(10, len(validate_dataset))):
        image, label, path = validate_dataset[i]
        filename = os.path.basename(path)
        class_name = idx_to_class.get(label, 'unknown')
        print(f"  {i + 1}. {filename}: label={label} ({class_name})")

    # Load CSV to verify
    print(f"\nVerifying against VALIDATE CSV:")
    validate_csv_data = {}
    with open(Config.VALIDATE_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '').strip()
            label_str = row.get('label', '').strip().lower()
            validate_csv_data[filename] = label_str

    # Check if first 10 match
    print(f"CSV labels for first 10 files:")
    validate_matches = 0
    for i in range(min(10, len(validate_dataset))):
        image, label, path = validate_dataset[i]
        filename = os.path.basename(path)
        csv_label = validate_csv_data.get(filename, 'NOT FOUND')
        dataset_label = idx_to_class.get(label, 'unknown')
        match = "✓" if csv_label == dataset_label else "✗"
        if csv_label == dataset_label:
            validate_matches += 1
        print(f"  {match} {filename}: CSV={csv_label}, Dataset={dataset_label}")

    print(f"\nVALIDATE Dataset: {validate_matches}/10 matches ✓")

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"TEST Dataset:     {test_matches}/10 matches")
    print(f"VALIDATE Dataset: {validate_matches}/10 matches")

    if test_matches == 10 and validate_matches == 10:
        print("\n✅ ALL TESTS PASSED! Labels are loading correctly!")
    else:
        print(f"\n⚠️  WARNING: Some labels don't match!")
        print(f"   Check your config.py CLASS_TO_IDX_4 mapping")

    print("=" * 70)


if __name__ == "__main__":
    test_data_loader()