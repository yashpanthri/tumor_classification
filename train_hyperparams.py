"""
Hyperparameter Tuning - Automated Experiment Runner
Runs all SGD experiments in sequence with different LR, momentum, and batch sizes.

Experiments:
  H1: LR=0.001, Momentum=0.0, Batch=8
  H2: LR=0.001, Momentum=0.9, Batch=8
  H3: LR=0.01,  Momentum=0.0, Batch=8
  H4: LR=0.01,  Momentum=0.9, Batch=8
  H5: LR=0.01 (scheduled), Momentum=0.0, Batch=8
  H6: LR=0.01 (scheduled), Momentum=0.9, Batch=8
  H7: LR=0.001, Momentum=0.5, Batch=8
  H8: LR=0.01,  Momentum=0.5, Batch=8
  H9: LR=0.001, Momentum=0.9, Batch=16
  H10: LR=0.001, Momentum=0.9, Batch=32

Optimizer: SGD (Stochastic Gradient Descent)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime

from config import Config
from data_loader import BrainTumorDataset
from Autoencoder import ConvAutoencoder

EXPERIMENTS = [
    {
        'name': 'H1_lr001_mom00_b8',
        'learning_rate': 0.001,
        'momentum': 0.0,
        'batch_size': 8,
        'use_scheduler': False
    },
    {
        'name': 'H2_lr001_mom09_b8',
        'learning_rate': 0.001,
        'momentum': 0.9,
        'batch_size': 8,
        'use_scheduler': False
    },
    {
        'name': 'H3_lr01_mom00_b8',
        'learning_rate': 0.01,
        'momentum': 0.0,
        'batch_size': 8,
        'use_scheduler': False
    },
    {
        'name': 'H4_lr01_mom09_b8',
        'learning_rate': 0.01,
        'momentum': 0.9,
        'batch_size': 8,
        'use_scheduler': False
    },
    {
        'name': 'H5_sched_mom00_b8',
        'learning_rate': 0.01,
        'momentum': 0.0,
        'batch_size': 8,
        'use_scheduler': True
    },
    {
        'name': 'H6_sched_mom09_b8',
        'learning_rate': 0.01,
        'momentum': 0.9,
        'batch_size': 8,
        'use_scheduler': True
    },
    {
        'name': 'H7_lr001_mom05_b8',
        'learning_rate': 0.001,
        'momentum': 0.5,
        'batch_size': 8,
        'use_scheduler': False
    },
    {
        'name': 'H8_lr01_mom05_b8',
        'learning_rate': 0.01,
        'momentum': 0.5,
        'batch_size': 8,
        'use_scheduler': False
    },
    {
        'name': 'H9_lr001_mom09_b16',
        'learning_rate': 0.001,
        'momentum': 0.9,
        'batch_size': 16,
        'use_scheduler': False
    },
    {
        'name': 'H10_lr001_mom09_b32',
        'learning_rate': 0.001,
        'momentum': 0.9,
        'batch_size': 32,
        'use_scheduler': False
    },
]

LATENT_DIM = 256
EPOCHS = 50
PATIENCE = 10
MIN_DELTA = 0.0001
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6
GRAD_CLIP_MAX = 1.0


def train_single_experiment(exp_config, train_dataset, device):
    """Train autoencoder with specified hyperparameters"""

    exp_name = exp_config['name']
    learning_rate = exp_config['learning_rate']
    momentum = exp_config['momentum']
    batch_size = exp_config['batch_size']
    use_scheduler = exp_config['use_scheduler']

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Momentum: {momentum}")
    print(f"   Batch Size: {batch_size}")
    print(f"   LR Scheduler: {'ON' if use_scheduler else 'OFF'}")
    print(f"   Gradient Clipping: {GRAD_CLIP_MAX}")
    print(f"   Latent Dim: {LATENT_DIM}")

    # Output directories
    output_dir = os.path.join(Config.BASE_DIR, f"outputs_{exp_name}")
    model_dir = os.path.join(output_dir, 'models')
    vis_dir = os.path.join(output_dir, 'visualizations')
    results_dir = os.path.join(output_dir, 'results')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"   Output: {output_dir}")

    # Create data loader with experiment's batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"   Batches/epoch: {len(train_loader)}")

    # Create fresh model
    model = ConvAutoencoder(latent_dim=LATENT_DIM)
    model = model.to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer - SGD
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    # Learning rate scheduler (optional) - FIXED: removed verbose parameter
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR
        )
        print(f"   Scheduler: ReduceLROnPlateau (patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR})")

    # Training loop
    train_losses = []
    learning_rates = []
    best_loss = float('inf')
    epochs_no_improve = 0

    start_time = datetime.now()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        nan_count = 0

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data[0].to(device)

            # Forward pass
            reconstructed, latent = model(images)
            loss = criterion(reconstructed, images)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"   WARNING: NaN/Inf loss at batch {batch_idx}!")
                if nan_count > 10:
                    print("   Error: Too many NaN losses. Stopping experiment.")
                    raise ValueError("Loss became NaN/Inf")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX)
            optimizer.step()

            epoch_loss += loss.item()

            # Progress update every 100 batches
            if batch_idx % 100 == 0:
                print(f"   Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f} LR: {current_lr:.6f}")

        # Average loss
        num_batches = len(train_loader) - nan_count
        if num_batches == 0:
            raise ValueError("All batches had NaN loss")
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        print(f"   Epoch {epoch + 1} Complete - Avg Loss: {avg_loss:.6f}")

        # Update scheduler
        if scheduler is not None:
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"   LR Reduced: {current_lr:.6f} -> {new_lr:.6f}")

        # Check improvement
        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            epochs_no_improve = 0

            # Save best model
            best_model_path = os.path.join(model_dir, 'autoencoder_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'latent_dim': LATENT_DIM,
                'hyperparams': exp_config
            }, best_model_path)
            print(f"   New best model! Loss: {best_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print(f"   No improvement for {epochs_no_improve} epochs")

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"   -> Early stopping at epoch {epoch + 1}")
            break

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(model_dir, f'autoencoder_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

    training_time = datetime.now() - start_time

    # Save plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'Training Loss\nBest: {best_loss:.6f}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(learning_rates[:len(train_losses)], 'r-', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title(f'Learning Rate\nMom: {momentum}, Batch: {batch_size}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.suptitle(f'{exp_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    loss_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save history - FIXED: using SCHEDULER_FACTOR instead of Scheduler
    history_path = os.path.join(results_dir, 'training_history.txt')
    with open(history_path, 'w', encoding='utf-8') as f:
        f.write(f"EXPERIMENT: {exp_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("HYPERPARAMETERS:\n")
        f.write(f"   Learning Rate: {learning_rate}\n")
        f.write(f"   Momentum: {momentum}\n")
        f.write(f"   Batch Size: {batch_size}\n")
        f.write(f"   LR Scheduler: {'ReduceLROnPlateau' if use_scheduler else 'Fixed'}\n")
        f.write(f"   Scheduler Patience: {SCHEDULER_PATIENCE}\n")
        f.write(f"   Scheduler Factor: {SCHEDULER_FACTOR}\n")
        f.write(f"   Gradient Clipping: {GRAD_CLIP_MAX}\n")
        f.write(f"   Latent Dim: {LATENT_DIM}\n\n")
        f.write("RESULTS:\n")
        f.write(f"   Epochs trained: {len(train_losses)}\n")
        f.write(f"   Best loss: {best_loss:.6f}\n")
        f.write(f"   Final loss: {train_losses[-1]:.6f}\n")
        f.write(f"   Training time: {training_time}\n\n")
        f.write("LOSS PER EPOCH:\n")
        for i, (loss, lr) in enumerate(zip(train_losses, learning_rates)):
            f.write(f"   Epoch {i + 1}: Loss={loss:.6f}, LR={lr:.6f}\n")

    # Return results
    return {
        'name': exp_name,
        'epochs': len(train_losses),
        'best_loss': best_loss,
        'final_loss': train_losses[-1],
        'training_time': str(training_time),
        'output_dir': output_dir
    }


def run_all_experiments():
    """Run all hyperparameter experiments in sequence"""

    print("=" * 70)
    print("HYPERPARAMETER TUNING - AUTOMATED RUNNER")
    print("=" * 70)
    print(f"\nTotal experiments: {len(EXPERIMENTS)}")
    print(f"Base directory: {Config.BASE_DIR}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (this will be slow!)")

    # Load dataset once
    print("\nLoading dataset...")
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = BrainTumorDataset(
        folder_path=Config.TRAIN_FOLDER,
        csv_path=None,
        transform=train_transform,
        load_labels=False
    )
    print(f"Loaded {len(train_dataset)} training images")

    # Run experiments
    all_results = []
    total_start = datetime.now()

    for i, exp_config in enumerate(EXPERIMENTS):
        print(f"\n{'#' * 70}")
        print(f"# STARTING EXPERIMENT {i + 1}/{len(EXPERIMENTS)}: {exp_config['name']}")
        print(f"{'#' * 70}")

        try:
            result = train_single_experiment(exp_config, train_dataset, device)
            all_results.append(result)

            print(f"\n   COMPLETED: {result['name']}")
            print(f"   Best Loss: {result['best_loss']:.6f}")
            print(f"   Time: {result['training_time']}")

        except Exception as e:
            print(f"\n   ERROR in {exp_config['name']}: {e}")
            all_results.append({
                'name': exp_config['name'],
                'error': str(e)
            })

    total_time = datetime.now() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {total_time}")

    # Results table
    print("\n" + "-" * 70)
    print(f"{'Experiment':<25} {'Epochs':<8} {'Best Loss':<12} {'Time':<15}")
    print("-" * 70)

    for r in all_results:
        if 'error' in r:
            print(f"{r['name']:<25} {'ERROR':<8} {r['error'][:30]}")
        else:
            print(f"{r['name']:<25} {r['epochs']:<8} {r['best_loss']:<12.6f} {r['training_time'][:12]}")

    print("-" * 70)

    # Save summary
    summary_path = os.path.join(Config.BASE_DIR, 'hyperparameter_tuning_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("HYPERPARAMETER TUNING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time}\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Experiment':<25} {'Epochs':<8} {'Best Loss':<12} {'Time':<15}\n")
        f.write("-" * 60 + "\n")

        for r in all_results:
            if 'error' in r:
                f.write(f"{r['name']:<25} ERROR: {r['error']}\n")
            else:
                f.write(f"{r['name']:<25} {r['epochs']:<8} {r['best_loss']:<12.6f} {r['training_time']}\n")

        f.write("-" * 60 + "\n")

        # Find best
        valid_results = [r for r in all_results if 'best_loss' in r]
        if valid_results:
            best = min(valid_results, key=lambda x: x['best_loss'])
            f.write(f"\nBEST MODEL: {best['name']}\n")
            f.write(f"   Best Loss: {best['best_loss']:.6f}\n")
            f.write(f"   Output: {best['output_dir']}\n")

    print(f"\nSummary saved: {summary_path}")

    # Print best model
    valid_results = [r for r in all_results if 'best_loss' in r]
    if valid_results:
        best = min(valid_results, key=lambda x: x['best_loss'])
        print(f"\n{'=' * 70}")
        print(f"BEST MODEL: {best['name']}")
        print(f"   Best Loss: {best['best_loss']:.6f}")
        print(f"   Output: {best['output_dir']}")
        print(f"{'=' * 70}")

        print(f"\nNEXT STEPS for best model:")
        print(f"1. Update config.py:")
        print(f"   RESULTS_DIR = r\"{best['output_dir']}\"")
        print(f"   MODEL_PATH = r\"{os.path.join(best['output_dir'], 'models', 'autoencoder_best.pth')}\"")
        print(f"   LATENT_DIM = 256")
        print(f"\n2. Run evaluation:")
        print(f"   python extract_features.py")
        print(f"   python k_means_clustering_k5.py")
        print(f"   python k5_evaluation.py")
        print(f"   python k5_evaluation_binary.py")


if __name__ == "__main__":
    run_all_experiments()