from pathlib import Path

import torch
from GenerativeAdverserialNetwork import get_generator, denorm, save_image


if __name__ == "__main__":

    
    level = 10
    generator_state_file = f"./generated/g_state_{level}.pth"
    discriminator_state_file = f"./generated/d_state_{level}.pth"

    generator_state_file = Path(generator_state_file)
    discriminator_state_file = Path(discriminator_state_file)

    latent_size = 128
    stats = ((.5, .5, .5), (.5, .5, .5))

    generator = get_generator(latent_size)
    generator.load_state_dict(torch.load(generator_state_file))

    latent = torch.randn(64, latent_size, 1, 1)
    images = generator(latent)

    image_full_name = f"./generated/test{level}.png"
    print(f"Saving image: {image_full_name}")
    save_image(denorm(images, stats), image_full_name, nrow=8)

