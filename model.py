
#This implementation is based upon three consecutive research papers,
#on Style GANs. This implemetation focuses on the styleGAN veersion 2
#which is state of the art of GAN models for generating images. 

#The paper is made possible by two previous research papers
#  "PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION" -> https://arxiv.org/pdf/1710.10196

#  "A Style-Based Generator Architecture for Generative Adversarial Networks" -> https://arxiv.org/pdf/1812.04948v3
    


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedWeight(nn.Module):

    def __init__(self, shape):

        super().__init__()

        self.c = 1 / np.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        """
        Multiply the weight by the constant c.

        Returns:
            torch.Tensor: The result of weight * c
        """
        return self.weight * self.c

class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = 0.):

        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        """
        Applies a linear transformation to the input tensor.

        Args:
            x: The input tensor. Should be a 2D tensor of shape (batch_size, in_features).

        Returns:
            The output tensor of shape (batch_size, out_features).
        """
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size, padding = 0):

        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        """
        Applies a 2D convolution over an input signal composed of several input planes.

        Args:
            x: The input tensor. Should be a 4D tensor of shape (batch_size, in_features, height, width).

        Returns:
            The output tensor of shape (batch_size, out_features, height, width).
        """
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

class MappingNetwork(nn.Module):

    def __init__(self, features: int, num_layers: int):

        super().__init__()


        layers = []
        for _ in range(num_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):

        # Normalize
        """
        Processes the input tensor through the mapping network after normalizing it.

        Args:
            z (torch.Tensor): The input tensor representing latent vectors. 
                              Should be a 2D tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: The output tensor after passing through the mapping network, 
                          of the same shape as the input.
        """

        z = F.normalize(z, dim=1)
        return self.net(z)


class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8):

        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):

        """
        Applies a 2D convolution over an input signal composed of several input planes.

        Args:
            x: The input tensor. Should be a 4D tensor of shape (batch_size, in_features, height, width).
            s: The style tensor. Should be a 2D tensor of shape (batch_size, features).

        Returns:
            The output tensor of shape (batch_size, out_features, height, width).
        """
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)

class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):

        """
        Applies the style block to the input tensor.

        Args:
            x: The input tensor. Should be a 4D tensor of shape (batch_size, in_features, height, width).
            w: The style tensor. Should be a 2D tensor of shape (batch_size, W_DIM).
            noise: The noise tensor. Should be a 4D tensor of shape (batch_size, 1, height, width).

        Returns:
            The output tensor of shape (batch_size, out_features, height, width).
        """
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])

class ToRGB(nn.Module):

    def __init__(self, W_DIM, features):

        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        """
        Applies the ToRGB layer transformations to the input tensor.

        Args:
            x: The input tensor. Should be a 4D tensor of shape (batch_size, features, height, width).
            w: The style tensor. Should be a 2D tensor of shape (batch_size, W_DIM).

        Returns:
            The output tensor of shape (batch_size, 3, height, width), representing the RGB image.
        """

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):

        """
        Applies the GeneratorBlock transformations to the input tensor.

        Args:
            x: The input tensor. Should be a 4D tensor of shape (batch_size, in_features, height, width).
            w: The style tensor. Should be a 2D tensor of shape (batch_size, W_DIM).
            noise: The noise tensor. Should be a list of two 4D tensors of shape (batch_size, 1, height, width),
                   where the first tensor represents the noise for the first style block and the second tensor
                   represents the noise for the second style block.

        Returns:
            A tuple of two tensors. The first tensor is the output tensor of the second style block, representing
            the feature map, and the second tensor is the output of the ToRGB layer, representing the RGB image.
        """
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb

class Generator(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])

        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):

        """
        Applies the Generator transformations to the input tensor.

        Args:
            w: The style tensor. Should be a 3D tensor of shape (n_styles, batch_size, W_DIM).
            input_noise: The noise tensor. Should be a list of n_styles tuples, where each tuple
                         contains two 4D tensors of shape (batch_size, 1, height, width), representing
                         the noise for the first style block and the second style block.

        Returns:
            The output tensor of shape (batch_size, 3, height, width), representing the RGB image.
        """
        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)

    

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        self.scale = 1 / np.sqrt(2)

    def forward(self, x):
        """
        Applies the DiscriminatorBlock transformations to the input tensor.

        Args:
            x: The input tensor. Should be a 4D tensor of shape (batch_size, in_features, height, width).

        Returns:
            The output tensor of shape (batch_size, out_features, height/2, width/2),
            after applying down-sampling, convolution, and residual connections.
        """

        residual = self.residual(x)

        x = self.block(x)

        
        x = self.down_sample(x)

        return (x + residual) * self.scale


class Discriminator(nn.Module):

    def __init__(self, log_resolution, n_features = 64, max_features = 256):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution)]

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)


    def minibatch_std(self, x):
        """
        Computes the minibatch standard deviation for the input tensor.

        Args:
            x: The input tensor. Should be a 4D tensor of shape (batch_size, features, height, width).

        Returns:
            The output tensor of shape (batch_size, features + 1, height, width), where the last feature
            dimension contains the minibatch standard deviation.
        """

        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x):

        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)
