"""
DCGAN Models: Generator and Discriminator
Implementation based on the DCGAN paper (Radford et al., 2015)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator
    Transforms random noise vector z ∈ ℝ^100 to 64x64 RGB image

    Architecture:
    - Input: z (100-dim noise vector)
    - Output: 3x64x64 RGB image, values in [-1, 1]
    """

    def __init__(self, nz=100, ngf=64, nc=3):
        """
        Args:
            nz: Size of latent z vector (default: 100)
            ngf: Size of feature maps in generator (default: 64)
            nc: Number of channels in output images (default: 3 for RGB)
        """
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Layer 1: z -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Layer 2: (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Layer 3: (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Layer 4: (ngf*2) x 16 x 16 -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Layer 5: ngf x 32 x 32 -> nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        Forward pass
        Args:
            input: Batch of noise vectors (batch_size, nz, 1, 1)
        Returns:
            Generated images (batch_size, nc, 64, 64)
        """
        return self.main(input)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator
    Classifies 64x64 RGB images as real or fake

    Architecture:
    - Input: 3x64x64 RGB image
    - Output: Scalar probability in [0, 1]
    """

    def __init__(self, nc=3, ndf=64):
        """
        Args:
            nc: Number of channels in input images (default: 3 for RGB)
            ndf: Size of feature maps in discriminator (default: 64)
        """
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Layer 1: nc x 64 x 64 -> ndf x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: ndf x 32 x 32 -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: (ndf*8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Forward pass
        Args:
            input: Batch of images (batch_size, nc, 64, 64)
        Returns:
            Probability that input is real (batch_size, 1, 1, 1)
        """
        return self.main(input)


def weights_init(m):
    """
    Custom weights initialization called on generator and discriminator
    From DCGAN paper: all weights initialized from Normal(0, 0.02)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
