
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from model import Generator, Discriminator


# -------------------------------
# 1. Settings
# -------------------------------

LATENT_DIM = 100
BATCH_SIZE = int(os.environ.get("DCGAN_BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("DCGAN_EPOCHS", 5))
LEARNING_RATE = float(
    os.environ.get("DCGAN_LEARNING_RATE", 0.0002)
)
DATASET_SIZE = int(os.environ.get("DCGAN_DATASET_SIZE", 2000))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("data", exist_ok=True)
os.makedirs("outputs/generated", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)


# -------------------------------
# 2. Load Fashion-MNIST
# -------------------------------

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Use a small subset to keep training manageable
    dataset = Subset(dataset, range(min(DATASET_SIZE, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print("Training images:", len(dataset))
    return loader


# -------------------------------
# 3. Save generated image samples
# -------------------------------

def save_generated_images(generator, epoch):
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)
        images = generator(noise).cpu()

    grid = torchvision.utils.make_grid(
        images, nrow=4, normalize=True, value_range=(-1, 1)
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"Generated Images - Epoch {epoch}")
    plt.savefig(f"outputs/generated/epoch_{epoch}.png")
    plt.close()

    generator.train()


# -------------------------------
# 4. Main training function
# -------------------------------

def train_model(epochs=EPOCHS):
    loader = load_dataset()

    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer_g = optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )

    loss_file = "outputs/losses.csv"

    # Create CSV with headers if it doesn't exist
    if not os.path.exists(loss_file):
        with open(loss_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "generator_loss", "discriminator_loss"])

    for epoch in range(1, epochs + 1):
        total_g_loss = 0.0
        total_d_loss = 0.0

        for real_images, _ in loader:
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # -----------------------
            # Train discriminator
            # -----------------------

            optimizer_d.zero_grad()

            real_output = discriminator(real_images)
            loss_real = criterion(real_output, real_labels)

            noise = torch.randn(
                batch_size, LATENT_DIM, 1, 1, device=DEVICE
            )
            fake_images = generator(noise)

            fake_output = discriminator(fake_images.detach())
            loss_fake = criterion(fake_output, fake_labels)

            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optimizer_d.step()

            # -----------------------
            # Train generator
            # -----------------------

            optimizer_g.zero_grad()

            fake_output = discriminator(fake_images)
            loss_g = criterion(fake_output, real_labels)

            loss_g.backward()
            optimizer_g.step()

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()

        avg_g_loss = total_g_loss / len(loader)
        avg_d_loss = total_d_loss / len(loader)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Generator Loss: {avg_g_loss:.4f} | "
            f"Discriminator Loss: {avg_d_loss:.4f}"
        )

        # Record losses
        with open(loss_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_g_loss, avg_d_loss])

        # Save generated samples
        save_generated_images(generator, epoch)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict()
        }

        torch.save(
            checkpoint,
            f"outputs/checkpoints/checkpoint_epoch_{epoch}.pth"
        )

    print("Training complete!")
    print("Generated images: outputs/generated/")
    print("Loss history: outputs/losses.csv")
    print("Checkpoints: outputs/checkpoints/")

    return generator, discriminator


if __name__ == "__main__":
    print("Using device:", DEVICE)
    train_model()
