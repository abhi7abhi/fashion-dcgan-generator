
import torch
import torch.nn as nn


# Generator: converts random noise into a 28x28 clothing image
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            # Input: (batch_size, 100, 1, 1)
            nn.ConvTranspose2d(
                latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Output: (batch_size, 64, 14, 14)
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Output: (batch_size, 1, 28, 28)
            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.network(noise)


# Discriminator: predicts whether an image is real or generated
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            # Input: (batch_size, 1, 28, 28)
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: (batch_size, 128, 7, 7)
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            # One output score for each image
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, image):
        return self.network(image)


# Test the model when this file is run directly
if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()

    # Create a batch of 4 random noise vectors
    noise = torch.randn(4, 100, 1, 1)

    # Generate fake clothing images
    fake_images = generator(noise)

    # Check if the discriminator can score those images
    scores = discriminator(fake_images)

    print("Generated image shape:", fake_images.shape)
    print("Discriminator output shape:", scores.shape)
