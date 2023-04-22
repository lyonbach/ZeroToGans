# import opendatasets as od
# od.download("https://www.kaggle.com/splcher/animefacedataset", data_dir="./data")


from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision.transforms as T


def denorm(image_tensors, stats):
    return image_tensors * stats[1][0] + stats[0][0]


def get_discriminator():

    return nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

    nn.Flatten(),
    nn.Sigmoid())


def get_generator(latent_size=128):

    return nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=2, padding=0, bias=True),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)


def train_discriminator(discriminator, generator, real_images, latent_size, opt_d, device="cpu"):

    # Clear discriminator gradients
    opt_d.zero_grad()
    real_images = real_images.to(device)

    # Pass real images through discriminator
    real_predictions = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_predictions, real_targets)
    real_score = torch.mean(real_predictions).item()

    # Generate fake images
    batch_size = len(real_images)
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_predictions = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_predictions, fake_targets)
    fake_score = 1 - torch.mean(fake_predictions).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()

    return loss.item(), real_score, fake_score


def train_generator(generator, discriminator, batch_size, latent_size, opt_g, device="cpu"):

    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    predictions = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(predictions, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


def save_samples(generator, idx, latent_tensors, output_directory, stats):

    fake_images = generator(latent_tensors)
    name = f"generated-{str(idx).zfill(4)}.png"

    output_directory = Path(output_directory)
    image_full_name = output_directory / name

    if not output_directory.is_dir():
        output_directory.mkdir()

    print(f"Saving image: {image_full_name}")
    save_image(denorm(fake_images, stats), image_full_name, nrow=8)


def fit(discriminator, generator, epochs, lr, latent_size, stats, train_dl, output_dir, start_idx=1, device="cpu"):

    # Keep losses and scores
    losses_g = []
    losses_d = []

    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = opt.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = opt.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

    for idx, epoch in enumerate(range(start_idx, start_idx + epochs), start=1):

        print(f"Starting epoch: {epoch}")
        for real_images, _ in tqdm(train_dl):
            loss_d, real_score, fake_score = train_discriminator(discriminator, generator, real_images, latent_size, opt_d, device=device)
            loss_g = train_generator(generator, discriminator, len(real_images), latent_size, opt_g, device=device)

        # Record losses and scores (last batch)
        losses_d.append(loss_d)
        losses_g.append(loss_g)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses and scores (last batch)
        print((f"Epoch [{idx}({epoch+1})/{epochs}], loss_g: {loss_g:.4f},"
               f"loss_d: {loss_d:.4f}, real_score: {real_score:.4f}, fake_score: {fake_score:.4f}"))

        # Save images for this batch
        save_samples(generator, epoch, fixed_latent, output_dir, stats)

        model_state_file_full_name = Path(output_dir) / f"g_state_{epoch}.pth"
        print(f"Saving model state at: {model_state_file_full_name}")
        torch.save(generator.state_dict(), model_state_file_full_name)

        model_state_file_full_name = Path(output_dir) / f"d_state_{epoch}.pth"
        print(f"Saving model state at: {model_state_file_full_name}")
        torch.save(discriminator.state_dict(), model_state_file_full_name)

    return losses_g, losses_d, real_scores, fake_scores


if __name__ == "__main__":

    image_size = 64
    batch_size = 128
    latent_size = 128
    stats = ((.5, .5, .5), (.5, .5, .5))
    lr = 0.0002
    epochs = 10

    data_dir = "./data/animefacedataset"
    output_dir = "./generated"
    train_ds = ImageFolder(data_dir, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)
    ]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3)

    generator = get_generator(latent_size=latent_size)
    generator.to(device)

    discriminator = get_discriminator()
    discriminator.to(device)

    history = fit(discriminator, generator, epochs, lr, latent_size, stats, train_dl, output_dir, device=device)
