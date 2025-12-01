import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # Layer 1 512x512x3 -> 256x256x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2 256x256x32 -> 128x128x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3 128x128x64 → 64x64x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 4: 64x64x128 → 32x32x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 5: 32x32x256 → 16x16x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.fc_encode = nn.Linear(16*16*512, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 16 * 16 * 512)

        self.decoder = nn.Sequential(
            # Layer 1 16x16x512 → 32x32x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 2 32x32x256 → 64x64x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 3 64x64x128 → 128x128x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 4 128x128x64 → 256x256x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 5: 256x256x32 → 512x512x3
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        # Pass through CNN Layers
        x = self.encoder(x)

        # Flatten to 1D vector
        x = x.view(x.size(0), -1)

        # Compress to latent space
        latent = self.fc_encode(x)
        return latent

    def decode(self, latent):
        # Expand from latent space
        x = self.fc_decode(latent)

        # Reshape to 4D tensor for CNN layers
        x = x.view(x.size(0), 512, 16, 16)

        # Pass through deconvolutional layers
        reconstructed = self.decoder(x)

        return reconstructed

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)

        return reconstructed, latent

def test_autoencoder():
    print("="*60)
    print('Testing Autoencoder')
    print("=" * 60)

    dummy_input = torch.randn(4, 3, 512, 512)
    print(f"\n1. Created dummy input: {dummy_input.shape}")

    model = ConvAutoencoder(latent_dim=128)
    print(f"2. Model CNN Autoencoder created!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n3. Model Parameters")
    print(f"Total Trainable Params: {trainable_params}")
    print(f"Total Trainable Params: {total_params}")

    # Test encoding
    print(f"\n4. Testing encoder...")
    latent = model.encode(dummy_input)
    print(f"   Input shape:  {dummy_input.shape}  (786,432 values per image)")
    print(f"   Latent shape: {latent.shape}  (128 values per image)")
    print(f"   Compression ratio: {786432 / 128:.1f}x")

    # Test decoding
    print(f"\n5. Testing decoder...")
    reconstructed = model.decode(latent)
    print(f"   Latent shape:        {latent.shape}")
    print(f"   Reconstructed shape: {reconstructed.shape}")

    # Test full forward pass
    print(f"\n6. Testing full forward pass...")
    reconstructed, latent = model(dummy_input)
    print(f"   Input shape:        {dummy_input.shape}")
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Latent shape:        {latent.shape}")

    # Check output range
    print(f"\n7. Checking output range (should be [0, 1])...")
    print(f"   Min value: {reconstructed.min().item():.4f}")
    print(f"   Max value: {reconstructed.max().item():.4f}")

    print("\n" + "=" * 70)
    print("AUTOENCODER ARCHITECTURE TEST COMPLETE")
    print("=" * 70)
    print("\nThe autoencoder is ready to use!")

if __name__ == '__main__':
    test_autoencoder()