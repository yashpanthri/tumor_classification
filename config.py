
import os
import torch


class Config:

    BASE_DIR = r"C:\Users\mlshe\IS579"

    TRAIN_FOLDER = os.path.join(BASE_DIR, "outputs_baseline", "svd_95", "Training")
    TEST_FOLDER = os.path.join(BASE_DIR,  "outputs_baseline", "svd_95", "Testing")
    VALIDATE_FOLDER = os.path.join(BASE_DIR,  "outputs_baseline", "svd_95", "Validation")

    # CSV annotation files (instead of CSV)
    TEST_CSV = os.path.join(BASE_DIR, "test_annotations1.csv")
    VALIDATE_CSV = os.path.join(BASE_DIR, "validate_annotations.csv")

    RESULTS_DIR = r"C:\Users\mlshe\IS579\outputs_SVD95_H9params"
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    CSV_DIR = os.path.join(RESULTS_DIR, "CSV")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    FEATURES_DIR = os.path.join(RESULTS_DIR, "features")

    MODEL_PATH = r"C:\Users\mlshe\IS579\outputs_SVD95_H9params\models\autoencoder_best.pth"

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU: {GPU_NAME} ({GPU_MEMORY:.2f} GB)")
    else:
        GPU_NAME = None
        GPU_MEMORY = 0
        print("⚠️  Using CPU")

    IMAGE_SIZE = 512
    CHANNELS = 3
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    LATENT_DIM = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 50

    USE_SCHEDULER = True
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.5

    USE_EARLY_STOPPING = True
    PATIENCE = 10
    MIN_DELTA = 0.0001

    SAVE_CHECKPOINTS = True
    CHECKPOINT_INTERVAL = 10

    N_CLUSTERS = 5
    KMEANS_N_INIT = 10
    KMEANS_MAX_ITER = 300
    KMEANS_RANDOM_STATE = 42

    SVD_VARIANCE_LEVELS = [0.90, 0.92, 0.95, 0.97, 0.99]
    SVD_PLOT_COMPONENTS = 50
    SVD_ENERGY_LEVEL = 0.99

    CLASS_NAMES_4 = {
        0: 'glioma',
        1: 'meningioma',
        2: 'pituitary',
        3: 'no_tumor'
    }

    CLASS_TO_IDX_4 = {v: k for k, v in CLASS_NAMES_4.items()}

    CLASS_NAMES_BINARY = {
        'glioma': 0,
        'meningioma': 1,
        'pituitary': 2,
        'no_tumor': 3
    }

    FOURCLASS_TO_BINARY = {
        0: 1,  # glioma → Tumor
        1: 1,  # meningioma → Tumor
        2: 1,  # pituitary → Tumor
        3: 0  # no_tumor → No_Tumor
    }

    FIG_SIZE_LARGE = (12, 8)
    FIG_SIZE_MEDIUM = (10, 6)
    FIG_SIZE_SMALL = (8, 6)
    FIG_DPI = 300

    COLORS_4CLASS = ['#2E7D32', '#1976D2', '#F57C00', '#C62828']
    COLORS_BINARY = ['#4CAF50', '#F44336']

    EXP_1_NAME = "Raw_Features"
    EXP_2_NAME = "SVD_Features"
    RESULTS_PREFIX = "brain_tumor_classification"

    CSV_COLUMNS = [
        'image_path',
        'true_label',
        'predicted_cluster',
        'predicted_label',
        'correct',
        'experiment'
    ]

    VERBOSE = 1
    RANDOM_SEED = 42
    NUM_WORKERS = 0
    PIN_MEMORY = True if torch.cuda.is_available() else False

    @classmethod
    def create_directories(cls):
        directories = [
            cls.RESULTS_DIR,
            cls.MODELS_DIR,
            cls.CSV_DIR,
            cls.PLOTS_DIR,
            cls.FEATURES_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            if cls.VERBOSE > 0:
                print(f"✓ Directory: {directory}")

    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Train: {cls.TRAIN_FOLDER}")
        print(f"Test: {cls.TEST_FOLDER}")
        print(f"Validate: {cls.VALIDATE_FOLDER}")
        print(f"Device: {cls.DEVICE}")
        print(f"Latent Dim: {cls.LATENT_DIM}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"K-Means Clusters: {cls.N_CLUSTERS}")
        print(f"SVD Variance Levels: {cls.SVD_VARIANCE_LEVELS}")
        print("=" * 70 + "\n")

    @classmethod
    def verify_paths(cls):
        print("\nVerifying paths...")

        paths = [
            ("Train", cls.TRAIN_FOLDER),
            ("Test", cls.TEST_FOLDER),
            ("Validate", cls.VALIDATE_FOLDER)
        ]

        all_exist = True
        for name, path in paths:
            if os.path.exists(path):
                print(f"✓ {name}: {path}")
            else:
                print(f"✗ {name}: {path} [NOT FOUND]")
                all_exist = False

        CSV_files = [
            ("Test CSV", cls.TEST_CSV),
            ("Validate CSV", cls.VALIDATE_CSV)
        ]

        for name, path in CSV_files:
            if os.path.exists(path):
                print(f"✓ {name}: {path}")
            else:
                print(f"⚠️  {name}: {path} [NOT FOUND]")

        print()
        return all_exist


# Run when imported or executed directly
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BRAIN TUMOR CLASSIFICATION - CONFIG TEST")
    print("=" * 70)

    Config.print_config()
    Config.verify_paths()
    Config.create_directories()

    print("✅ Configuration loaded successfully!")