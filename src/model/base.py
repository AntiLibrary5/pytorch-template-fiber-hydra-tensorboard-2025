import torch.nn as nn


class UNet(nn.Module):
    """A minimal autoencoder for image reconstruction demonstrations.
    """

    def __init__(self, in_channels=1, latent_dim=16):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            latent_dim: Number of feature maps in the bottleneck layer
        """
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: reduce spatial dimensions by 2, increase channels to 8
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            # Layer 2: further reduce spatial dimensions by 2, increase channels to latent_dim
            nn.Conv2d(8, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: start upsampling, reduce channels from latent_dim to 8
            nn.ConvTranspose2d(latent_dim, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            # Layer 2: final upsampling back to original dimensions
            nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Constrains output to [0,1] range for images
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        """Get the latent representation (encoding) of the input."""
        return self.encoder(x)

    def decode(self, x):
        """Reconstruct from the latent representation."""
        return self.decoder(x)