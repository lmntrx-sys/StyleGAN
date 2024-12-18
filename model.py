
#This implementation is based upon three consecutive research papers,
#on Style GANs. This implemetation focuses on the styleGAN veersion 2
#which is state of the art of GAN models for generating images. 

#The paper is made possible by two previous research papers
#  "PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION" -> https://arxiv.org/pdf/1710.10196

#  "A Style-Based Generator Architecture for Generative Adversarial Networks" -> https://arxiv.org/pdf/1812.04948v3
    


import numpy as np
import torch
from typing import Optional, Tuple

import torch.nn as nn
import math
import torch.nn.functional as F

import torch
import torch.nn as nn

class ExampleModule(nn.Module):
    def __init__(self, shape):
        super(ExampleModule, self).__init__()
        # Ensure shape is a tuple of ints
        if not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple")
        if not all(isinstance(dim, int) for dim in shape):
            shape = tuple(int(dim) for dim in shape)  # Convert to int if needed
        self.weight = nn.Parameter(torch.randn(shape))

# Example usage
shape = (3, 4)  # Valid shape
model = ExampleModule(shape)
print(model.weight)


class EqualizedWeights(nn.Module):

    def __init__(self, shape: Tuple[int, ...]):
        """
        Initialize the EqualizedWeights module.

        Args:
            shape (List[int]): Shape of the weight tensor.

        Attributes:
            weight (nn.Parameter): Learnable weight tensor.
            c (float): Constant value used in the forward method.

        """
        super().__init__()


        self.weight = nn.Parameter(torch.randn(shape))
        self.c = 1 / math.sqrt(np.prod(shape[1:]))


    def forward(self):
        
        return self.weight * self.c
    
class EqualizedLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):

        """
        Initialize the EqualizedLinear module.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Attributes:
            weight (EqualizedWeights): Learnable weight tensor.
            bias (nn.Parameter): Learnable bias parameter.

        """
        super().__init__()
        self.weight = EqualizedWeights([in_features, out_features])
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor):
        # Apply linear transformation
        return F.linear(x, self.weight(), self.bias)
    
class MappingNetwork(nn.Module):

    def __inti__(self, features: int, num_layers: int):

        """
        Initialize the MappingNetwork module.

        Args:
            features (int): Number of features in the network.
            num_layers (int): Number of layers in the network.

        """
        super().__init__()

        
        layers = []
        for _ in range(num_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        """
        Perform a forward pass through the MappingNetwork.

        This function takes an input tensor `x` and applies the following operations:
        - Normalizes `x` along dimension 1 using `F.normalize`.
        - Applies the network defined by `self.net` to the normalized input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the network.

        """
        # Normalize
        z = F.normalize(x, dim=1)
        return self.net(z)
    
class Conv2dModulate(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, demodulate: bool = True, eps: float = 1e-8):
        super().__init__()
        self.demodulate = demodulate
        self.out_ch = out_ch
        self.eps = eps

        self.padding = (kernel_size - 1 // 2)
        self.weight = EqualizedWeights([in_ch, out_ch, kernel_size, kernel_size])

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        # [b, c, h, w]
        b, c, h, w = x.shape
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :] * s

        if self.demodulate:
            demod = torch.rsqrt((weights**2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * demod

        # [b, c, h, w] -> [1, c, h, w]
        x = x.view(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.view(b * self.out_ch, c // self.out_ch, *ws)
        x = F.conv2d(x, weights, padding=self.padding, groups=b * self.out_ch)
        return x.reshape(b, self.out_ch, h, w)



class StyleBlock(nn.Module):

    def __init__(self, d_model, in_features, out_features):
        """
        Initialize the StyleBlock module.

        Args:
            d_model (int): Input feature dimension of the module.
            in_features (int): Input feature dimension of the style network.
            out_features (int): Output feature dimension of the style network.

        Attributes:
            to_style (EqualizedLinear): Style network.
            modulate (Conv2dModulate): Modulation layer.
            bias (nn.Parameter): Learnable bias parameter.
            scale_noise (nn.Parameter): Learnable noise scaling parameter.
            activation (nn.LeakyReLU): Activation function.

        """
        super().__init__()

        self.to_style = EqualizedLinear(in_features, out_features)
        self.modulate = Conv2dModulate(d_model, d_model, kernel_size=3)

        # learnable bias
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scale_noise = nn.Parameter(torch.randn(1))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):

        s = self.to_style(w)

        x = self.modulate(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise

        return self.activation(x + self.bias[None, :, None, None])

class ToRGB(nn.Module):

    def __init__(self, d_model, out_features):
        """
        Initialize the ToRGB module.

        Args:
            d_model (int): Input feature dimension of the module.
            out_features (int): Output feature dimension of the module.

        Attributes:
            to_rgb (Conv2dModulate): RGB synthesis layer.
            bias (nn.Parameter): Learnable bias parameter.
            activation (nn.LeakyReLU): Activation function.
            to_style (EqualizedLinear): Style network.

        """
        super().__init__()

        self.to_rgb = Conv2dModulate(out_features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.randn(3))

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.to_style = EqualizedLinear(d_model, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor):

        s = self.to_style(w)
        x = self.to_rgb(x, s)
        return self.activation(x + self.bias[None, :, None, None])
    


class GenertorBlock(nn.Module):

    def __init__(self, d_model, in_features, out_features):
        """
        Initialize a GeneratorBlock.

        Args:
            d_model (int): Input features to the style network.
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Attributes:
            to_rgb (ToRGB): RGB synthesis layer.
            to_style_block1 (StyleBlock): First style block.
            to_style_block2 (StyleBlock): Second style block.

        """
        super().__init__()

        self.to_rgb = ToRGB(d_model, out_features)
        # two style blocks
        self.to_style_block1 = StyleBlock(d_model, in_features, out_features)
        self.to_style_block2 = StyleBlock(d_model, out_features, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):

        # input into the first style block
        x = self.to_style_block1(x, w, noise[0])

        # input into the second style block
        x = self.to_style_block2(x, w, noise[1])
        rgb = self.to_rgb(x, w)


        return x, rgb
    
class Generator(nn.Module):

    def __init__(self, log_resolution: int, d_model: int , max_features: int = 512, n_features: int = 32 ):
        """
        Initialize the Generator module.

        Args:
            log_resolution (int): Logarithmic resolution to determine feature sizes.
            d_model: Dimension of the model.
            max_features (int, optional): Maximum number of features for any layer. Defaults to 512.
            n_features (int, optional): Number of features in a block. Defaults to 32.

        Attributes:
            initial_constant (nn.Parameter): Initial constant tensor for the generator.
            n_blocks (int): Number of generator blocks.
            to_style (StyleBlock): Initial style block.
            to_rgb (ToRGB): Initial RGB synthesis layer.
            blocks (nn.Sequential): Sequence of GeneratorBlock layers.
            upsample (nn.Upsample): Upsampling layer with a scale factor of 2.
        """
        super().__init__()

        features = [min(max_features, n_features, 2**i) for i in range(log_resolution, -2, -1)]
        self.initial_constant = nn.Parameter(torch.randn(1, features[0], 4, 4))

        # number of block 
        self.n_blocks = len(features) - 1

        self.to_style = StyleBlock(d_model, features[0], features[0])
        self.to_rgb = ToRGB(d_model, features[0])

        # generator blocks
        blocks = [GenertorBlock(d_model, features[i], features[i+1]) for i in range(self.n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):

        # batch size
        batch_size = w.shape[1]

        # initial constant
        x = self.initial_constant.expand(batch_size, 1, 1, 1)

        x = self.to_style(x, w[0], noise[0])

        
        for i in range(1, self.n_blocks):
            # upsampling
            x = self.upsample(x)
            x, rgb = self.blocks[i-1](x, w[i], noise[i])
            x = self.upsample(rgb)

        x = self.to_rgb(x, w[-1])

        return x


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        # residual
        self.residual = nn.Sequential(nn.AvgPool2d(2, stride=2),
                                      EqualizedLinear(in_features, out_features, kernel_size=1))
        
        # two convolutional layers
        self.blocks = nn.Sequential(EqualizedLinear(in_features, out_features, kernel_size=3, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    EqualizedLinear(out_features, out_features, kernel_size=3, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        self.downsample = nn.AvgPool2d(2, stride=2)

        self.scale = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor):

        res = self.residual(x)
        x = self.blocks(x)
        x = self.downsample(x)
        return self.scale * (x + res)

class Discriminator(nn.Module):

    def __init__(self, log_resolution, n_features = 64, max_features = 256):
        super().__init__()
        
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        self.from_rgb = nn.Sequential(
            Conv2dModulate(3, n_features, 1),
            nn.LeakyReLU(0.2, True)
        )

        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        self.conv = Conv2dModulate(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def minibatch_std(self, x):
        batch_stats = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        )

        return torch.cat([x, batch_stats], dim=1)

    def forward(self, x):

        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.minibatch_std(x)

        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)

        return self.final(x)