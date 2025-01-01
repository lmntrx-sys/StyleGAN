from model import MappingNetwork, Discriminator, Generator
from loss_penalty import get_w, get_noise, gradient_penalty, PathLengthPenalty
from loss_penalty import generate_examples, get_loader

import tqdm
import torch

EPOCHS = 100
LAMBDA_GP = 10
LOG_RESOLUTION = 7
W_DIM = 256
Z_DIM = 256
LEARNING_RATE = 1e-3

mapping_network = MappingNetwork()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def training(
        critic,
    gen,
    path_length_penalty,
    loader,
    opt_critic,
    opt_gen,
    opt_mapping_network
):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size)

        noise = get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())

            critic_real = critic(real)
            gp = gradient_penalty(critic, real, fake, DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

loader = get_loader()

gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
critic              = Discriminator(LOG_RESOLUTION).to(DEVICE)
mapping_network     = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

opt_gen             = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic          = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_mapping_network = torch.optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

gen.train()
critic.train()
mapping_network.train()

loader = get_loader()

for epoch in range(EPOCHS):
    training(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )
    if epoch % 50 == 0:
    	generate_examples(gen, epoch)