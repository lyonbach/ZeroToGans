import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.transforms as T

from GenerativeAdverserialNetwork import get_generator, get_discriminator, fit


if __name__ == "__main__":

    image_size = 64
    batch_size = 128
    latent_size = 128
    stats = ((.5, .5, .5), (.5, .5, .5))
    lr = 0.00005
    epochs = 50

    data_dir = "./data/animefacedataset"
    output_dir = "./generated"
    train_ds = ImageFolder(data_dir, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)
    ]))

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_idx = 15
    generator_state_file = f"./generated/g_state_{start_idx-1}.pth"
    print(f"Using generator state at: {generator_state_file}")
    generator = get_generator(latent_size=latent_size)
    generator.to(device)
    generator.load_state_dict(torch.load(generator_state_file))
    
    discriminator_state_file = f"./generated/d_state_{start_idx-1}.pth"
    print(f"Using discriminator state at: {discriminator_state_file}")
    discriminator = get_discriminator()
    discriminator.to(device)
    discriminator.load_state_dict(torch.load(discriminator_state_file))

    history = fit(discriminator, generator, epochs, lr, latent_size, stats, train_dl, output_dir, start_idx, device)
    