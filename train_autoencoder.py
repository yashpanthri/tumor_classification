import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from config import Config
from data_loader import BrainTumorDataset
from Autoencoder import ConvAutoencoder

class TrainingConfig:
    # Hyperparameters
    epoch = Config.EPOCHS
    batch_size = Config.BATCH_SIZE
    learning_rate = Config.LEARNING_RATE

    # Model Architecture
    latent_dim = Config.LATENT_DIM

    # Check to ensure overfitting is restricted
    early_stopping_patience = Config.PATIENCE
    min_delta = Config.MIN_DELTA

    # Save settings
    check_point_save = 5
    epoch_visual = 5

    # Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    output_dir = Config.RESULTS_DIR
    model_dir = os.path.join(output_dir, 'models')
    vis_dir = os.path.join(output_dir, 'visualizations')
    results_dir = os.path.join(output_dir, 'results')

def visualization_reconstruction(model, dataloader, device, epoch, save_path):

    model.eval()
    with torch.no_grad():
        batch_data = next(iter(dataloader))
        images = batch_data[0]
        images = images.to(device)

        model_output = model(images)
        reconstructed = model_output[0]

        images = images.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()

    n_images = min(4, images.shape[0])
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 4, 8))

    for i in range(n_images):
        # Show Original Image
        img = np.transpose(images[i], (1, 2, 0))
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original Image {i + 1}')
        axes[0, i].axis('off')

        # Show reconstructed image
        recon_image = np.transpose(reconstructed[i], (1, 2, 0))
        axes[1, i].imshow(recon_image)
        axes[1, i].set_title(f'Reconstructed Image {i + 1}')
        axes[1, i].axis('off')
    plt.suptitle(f'Epoch {epoch}: Original vs Reconstructed Images', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Visualization: {save_path}")

def plot_loss_curve(train_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', linewidth=1, markersize=6)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Reconstruction Loss (MSE)", fontsize=10)
    plt.title("Autoencoder Training Progress", fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Loss Curve: {save_path}")

def train_autoencoder():
    print("="*60)
    print("Training Autoencoder...")
    print("=" * 60)

    # Creating directories
    print("Creating Output Directories")
    os.makedirs(TrainingConfig.model_dir, exist_ok=True)
    os.makedirs(TrainingConfig.vis_dir,exist_ok=True)
    os.makedirs(TrainingConfig.results_dir,exist_ok=True)

    # General check on location of each output to location
    print(f"Models will be saved to: {TrainingConfig.model_dir}")
    print(f"Visualizations will be saved to: {TrainingConfig.vis_dir}")
    print(f"Results will be saved to: {TrainingConfig.results_dir}")

    # Device check
    device = TrainingConfig.device
    if device.type == 'cuda':
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"Cuda version: {torch.version.cuda}")

    else:
        print(f"No GPU available, using CPU instead")

    # Loading data set
    print("="*60)
    print("Loading Dataset...")
    print("=" * 60)

    try:
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        train_dataset = BrainTumorDataset(folder_path=Config.TRAIN_FOLDER,
                                          csv_path=None, transform=train_transform, load_labels=False,)

        train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

        print(f"Loaded {len(train_dataset)} training images")
        print(f"Batch Size: {TrainingConfig.batch_size}")
        print(f"Number of batches: {len(train_loader)}")

    except Exception as error:
        print(f"Error loading dataset: {error}")
        return

    print("Creating the Model...")
    model = ConvAutoencoder(latent_dim=TrainingConfig.latent_dim)
    model = model.to(device)

    # Count number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created with laten_dim: {TrainingConfig.latent_dim}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Compression ratio: {786432 / TrainingConfig.latent_dim:.1f}x")

    # Loss function setup
    print(f"Setting up Training Components")
    criterion = nn.MSELoss()
    print("Loss Function: MSE!")

    # Optimizer: Adam an adaptive learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=TrainingConfig.learning_rate, momentum=0.9)
    print("Optimizer Function: Stochastic Gradient Descent!")
    print(f"Learning Rate: {TrainingConfig.learning_rate}")

    print("Beginning Training...")

    train_losses = []
    best_loss = float('inf')
    epoch_no_improve = 0

    for epoch in range(TrainingConfig.epoch):
        model.train()
        epoch_loss = 0

        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data[0]
            labels = batch_data[1]

            images = images.to(device)

            model_output = model(images)
            reconstructed = model_output[0]
            latent = model_output[1]

            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

            # Print loss every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{TrainingConfig.epoch}"
                      f"Batch [{batch_idx}/{len(train_loader)}]"
                      f"Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{TrainingConfig.epoch} Complete!")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"{'=' * 70}\n")

        if avg_loss < best_loss - TrainingConfig.min_delta:
            best_loss = avg_loss
            epoch_no_improve = 0

            # Save best model
            best_model_path = os.path.join(
                TrainingConfig.model_dir,
                'autoencoder_best.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'latent_dim': TrainingConfig.latent_dim,
            }, best_model_path)

            print(f"   ✓ New best model! Saved to: {best_model_path}")
        else:
            epoch_no_improve += 1
            print(f"   No improvement for {epoch_no_improve} epochs")

        # Early stopping check

        if epoch_no_improve >= TrainingConfig.early_stopping_patience:
            print(f"Early stopping triggered!")
            print(f"No improvement for {TrainingConfig.early_stopping_patience} epochs")
            print(f"Best loss: {best_loss:.6f}")
            break

        if (epoch + 1) % TrainingConfig.check_point_save == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(
                TrainingConfig.model_dir,
                f'autoencoder_epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'latent_dim': TrainingConfig.latent_dim,
            }, checkpoint_path)
            print(f"   ✓ Saved checkpoint: {checkpoint_path}")

        if (epoch + 1) % TrainingConfig.epoch_visual == 0:
            # Create visualization
            vis_path = os.path.join(
                TrainingConfig.vis_dir,
                f'reconstruction_epoch_{epoch + 1}.png'
            )
            visualization_reconstruction(model, train_loader, device, epoch + 1, vis_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    # Plot final loss curve
    loss_curve_path = os.path.join(
        TrainingConfig.results_dir,
        'training_loss_curve.png'
    )
    plot_loss_curve(train_losses, loss_curve_path)

    # Final reconstruction visualization
    final_vis_path = os.path.join(
        TrainingConfig.vis_dir,
        'reconstruction_final.png'
    )
    visualization_reconstruction(model, train_loader, device, epoch + 1, final_vis_path)

    # Save training history
    history_path = os.path.join(
        TrainingConfig.results_dir,
        'training_history.txt'
    )
    with open(history_path, 'w') as f:
        f.write("AUTOENCODER TRAINING HISTORY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {len(train_losses)}\n")
        f.write(f"Best Loss: {best_loss:.6f}\n")
        f.write(f"Final Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Latent Dimension: {TrainingConfig.latent_dim}\n")
        f.write(f"Batch Size: {TrainingConfig.batch_size}\n")
        f.write(f"Learning Rate: {TrainingConfig.learning_rate}\n")
        f.write(f"Training Images: {len(train_dataset)}\n")
        f.write(f"\nLoss per epoch:\n")
        for i, loss in enumerate(train_losses):
            f.write(f"Epoch {i + 1}: {loss:.6f}\n")

    print(f"\n✓ Training history saved: {history_path}")
    print(f"\n{'=' * 70}")
    print("STAGE 4 COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Training Summary:")
    print(f"   Total epochs: {len(train_losses)}")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Final loss: {train_losses[-1]:.6f}")
    print(f"   Improvement: {(1 - train_losses[-1] / train_losses[0]) * 100:.1f}%")
    print(f"\n📁 Saved Files:")
    print(f"   Best model: {TrainingConfig.model_dir}/autoencoder_best.pth")
    print(f"   Loss curve: {loss_curve_path}")
    print(f"   Visualizations: {TrainingConfig.vis_dir}/")
    print(f"   Training history: {history_path}")
    print(f"\n🚀 Next Step: Stage 5 - Extract Latent Features")

if __name__ == "__main__":
    train_autoencoder()








