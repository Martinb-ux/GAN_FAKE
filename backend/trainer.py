"""
DCGAN Training Logic
Implements the alternating optimization algorithm for GAN training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from models import Generator, Discriminator, weights_init
import io
import base64
from PIL import Image


class DCGANTrainer:
    """
    Trainer class for DCGAN
    Manages training loop, optimization, and metrics
    """

    def __init__(self, device='cpu', nz=100, ngf=64, ndf=64, nc=3, lr=0.0002, beta1=0.5):
        """
        Initialize DCGAN trainer

        Args:
            device: 'cuda' or 'cpu'
            nz: Latent vector size
            ngf: Generator feature map size
            ndf: Discriminator feature map size
            nc: Number of image channels
            lr: Learning rate
            beta1: Beta1 for Adam optimizer
        """
        self.device = device
        self.nz = nz
        self.nc = nc

        # Initialize networks
        self.netG = Generator(nz, ngf, nc).to(device)
        self.netD = Discriminator(nc, ndf).to(device)

        # Apply weight initialization
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers (Adam with beta1=0.5 as per DCGAN paper)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

        # Fixed noise for visualization
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Training state
        self.is_training = False
        self.current_epoch = 0
        self.metrics = {
            'g_losses': [],
            'd_losses': [],
            'real_scores': [],
            'fake_scores': []
        }

    def get_dataloader(self, dataset_name='mnist', batch_size=64, num_workers=0):
        """
        Create dataloader for training

        Args:
            dataset_name: 'mnist' or 'fashion_mnist'
            batch_size: Batch size for training
            num_workers: Number of workers for data loading

        Returns:
            DataLoader object
        """
        # Custom transform to convert grayscale to RGB
        class ToRGB:
            def __call__(self, x):
                return x.repeat(3, 1, 1) if x.size(0) == 1 else x

        # Image transformations
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ToRGB()  # Convert grayscale to RGB
        ])

        if dataset_name == 'mnist':
            dataset = datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )
        elif dataset_name == 'fashion_mnist':
            dataset = datasets.FashionMNIST(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return dataloader

    def train_step(self, real_images):
        """
        Single training step - update D and G

        Args:
            real_images: Batch of real images

        Returns:
            Dictionary with losses and scores
        """
        batch_size = real_images.size(0)
        real_label = 1.0
        fake_label = 0.0

        # =====================
        # Update Discriminator
        # Maximize log(D(x)) + log(1 - D(G(z)))
        # =====================
        self.netD.zero_grad()

        # Train with real images
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
        output = self.netD(real_images).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake images
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake = self.netG(noise)
        label.fill_(fake_label)
        output = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Update D
        errD = errD_real + errD_fake
        self.optimizerD.step()

        # =====================
        # Update Generator
        # Maximize log(D(G(z))) (non-saturating loss)
        # =====================
        self.netG.zero_grad()
        label.fill_(real_label)
        output = self.netD(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        self.optimizerG.step()

        return {
            'loss_d': errD.item(),
            'loss_g': errG.item(),
            'real_score': D_x,
            'fake_score': D_G_z1,
            'fake_score_after_g': D_G_z2
        }

    def generate_images(self, num_images=64, noise=None):
        """
        Generate synthetic images

        Args:
            num_images: Number of images to generate
            noise: Optional noise vector (if None, random noise is used)

        Returns:
            Generated images tensor
        """
        self.netG.eval()
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(num_images, self.nz, 1, 1, device=self.device)
            fake_images = self.netG(noise)
        self.netG.train()
        return fake_images

    def images_to_base64(self, images, nrow=8):
        """
        Convert image tensor to base64 encoded string

        Args:
            images: Tensor of images (N, C, H, W)
            nrow: Number of images per row in grid

        Returns:
            Base64 encoded PNG image
        """
        from torchvision.utils import make_grid

        # Denormalize from [-1, 1] to [0, 1]
        images = images * 0.5 + 0.5
        images = torch.clamp(images, 0, 1)

        # Create grid
        grid = make_grid(images, nrow=nrow, padding=2)

        # Convert to PIL Image
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    def get_metrics(self):
        """Get current training metrics"""
        return {
            'epoch': self.current_epoch,
            'g_losses': self.metrics['g_losses'],
            'd_losses': self.metrics['d_losses'],
            'real_scores': self.metrics['real_scores'],
            'fake_scores': self.metrics['fake_scores']
        }

    def save_checkpoint(self, path='checkpoint.pth'):
        """Save model checkpoint"""
        torch.save({
            'epoch': self.current_epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'metrics': self.metrics
        }, path)

    def load_checkpoint(self, path='checkpoint.pth'):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.metrics = checkpoint['metrics']
