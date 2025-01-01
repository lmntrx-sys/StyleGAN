import math
import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
LOG_RESOLUTION = 7 
W_DIM = 256
from model import MappingNetwork

class PathLengthPenalty(nn.Module):

    def __init__(self, beta):

        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):

        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / math.sqrt(image_size)
        math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mapping_network = MappingNetwork()

def gradient_penalty(critic, real, fake, cevice='cpu'):

    B, c, h, w = real.shape
    beta = torch.rand((B, 1, 1, 1)).repeat(1, c, h, w).to(device=DEVICE)

    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate the critic score
    mixed_scores = critic(interpolated_images)

    # Take the gradient of hte scores with respect to the images

    grad = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True, 
        retain_graph=True,
    )[0]

    grad = grad.view(grad.shape[0], -1)
    grad_norm = grad.norm(2, dim=1)
    return torch.mean((grad_norm - 1) ** 2)




def get_w(batch_size):

    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)



def get_noise(batch_size):

    noise = []
    resolution = 4

    for i in range(LOG_RESOLUTION):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

        noise.append((n1, n2))

        resolution *= 2

    return noise


def generate_examples(gen, epoch, n=100):
    
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    gen.train()