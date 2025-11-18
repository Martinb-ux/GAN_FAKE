"""
Diagnostic script to identify issues with the DCGAN backend
Run this to check what's causing errors
"""

import sys
import traceback


def check_imports():
    """Check if all required packages are installed"""
    print("Checking Python packages...")

    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }

    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✓ All packages installed\n")
    return True


def check_pytorch():
    """Check PyTorch installation and GPU availability"""
    print("Checking PyTorch...")

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")

        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available():
            print(f"  Apple Silicon GPU (MPS): ✓ Available")
            print(f"  Device: Apple M-series GPU")
        else:
            print(f"  Apple Silicon GPU (MPS): Not available")

        # Check for CUDA
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Determine which device will be used
        if torch.backends.mps.is_available():
            print("  → Will use: Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            print("  → Will use: NVIDIA GPU (CUDA)")
        else:
            print("  → Will use: CPU (training will be slower)")

        print("✓ PyTorch working\n")
        return True
    except Exception as e:
        print(f"✗ PyTorch error: {e}\n")
        return False


def test_models():
    """Test model creation"""
    print("Testing model creation...")

    try:
        from models import Generator, Discriminator, weights_init
        import torch

        # Test Generator
        print("  Creating Generator...")
        netG = Generator(nz=100, ngf=64, nc=3)
        netG.apply(weights_init)

        # Test forward pass
        test_noise = torch.randn(1, 100, 1, 1)
        output = netG(test_noise)
        print(f"    Input: {test_noise.shape}")
        print(f"    Output: {output.shape}")

        if output.shape != (1, 3, 64, 64):
            print(f"  ✗ Wrong output shape: {output.shape}")
            return False

        print("  ✓ Generator OK")

        # Test Discriminator
        print("  Creating Discriminator...")
        netD = Discriminator(nc=3, ndf=64)
        netD.apply(weights_init)

        # Test forward pass
        pred = netD(output)
        print(f"    Input: {output.shape}")
        print(f"    Output: {pred.shape}")

        if pred.shape != (1, 1, 1, 1):
            print(f"  ✗ Wrong output shape: {pred.shape}")
            return False

        print("  ✓ Discriminator OK")
        print("✓ Models working\n")
        return True

    except Exception as e:
        print(f"✗ Model error: {e}")
        traceback.print_exc()
        return False


def test_trainer():
    """Test trainer initialization"""
    print("Testing DCGANTrainer...")

    try:
        from trainer import DCGANTrainer
        import torch

        # Use MPS if available, otherwise CPU for testing
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        trainer = DCGANTrainer(device=device)

        print(f"  Device: {trainer.device}")
        print(f"  Latent size: {trainer.nz}")

        # Test generation
        print("  Testing image generation...")
        fake_images = trainer.generate_images(num_images=4)
        print(f"    Generated shape: {fake_images.shape}")

        # Test base64 conversion
        print("  Testing base64 conversion...")
        img_b64 = trainer.images_to_base64(fake_images, nrow=2)
        print(f"    Base64 length: {len(img_b64)} chars")

        print("✓ Trainer working\n")
        return True

    except Exception as e:
        print(f"✗ Trainer error: {e}")
        traceback.print_exc()
        return False


def test_dataloader():
    """Test dataloader creation"""
    print("Testing dataloader...")

    try:
        from trainer import DCGANTrainer
        import torch

        # Use MPS if available, otherwise CPU for testing
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        trainer = DCGANTrainer(device=device)

        print("  Creating MNIST dataloader...")
        print("  (This will download MNIST if not present, ~10MB)")

        dataloader = trainer.get_dataloader(
            dataset_name='mnist',
            batch_size=16,
            num_workers=0  # Use 0 to avoid multiprocessing issues
        )

        print(f"  Dataset size: {len(dataloader.dataset)}")
        print(f"  Batches: {len(dataloader)}")

        # Get one batch
        print("  Loading one batch...")
        images, labels = next(iter(dataloader))

        print(f"    Images shape: {images.shape}")
        print(f"    Labels shape: {labels.shape}")
        print(f"    Image dtype: {images.dtype}")
        print(f"    Image range: [{images.min():.3f}, {images.max():.3f}]")

        if images.shape[1] != 3:
            print(f"  ✗ Wrong number of channels: {images.shape[1]} (expected 3)")
            return False

        if images.shape[2] != 64 or images.shape[3] != 64:
            print(f"  ✗ Wrong image size: {images.shape[2]}x{images.shape[3]} (expected 64x64)")
            return False

        print("✓ Dataloader working\n")
        return True

    except Exception as e:
        print(f"✗ Dataloader error: {e}")
        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step"""
    print("Testing training step...")

    try:
        from trainer import DCGANTrainer
        import torch

        # Use MPS if available, otherwise CPU for testing
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        trainer = DCGANTrainer(device=device)

        # Create fake batch of real images
        real_images = torch.randn(4, 3, 64, 64, device=device) * 0.5

        print("  Running train_step...")
        metrics = trainer.train_step(real_images)

        print(f"    D loss: {metrics['loss_d']:.4f}")
        print(f"    G loss: {metrics['loss_g']:.4f}")
        print(f"    D(real): {metrics['real_score']:.4f}")
        print(f"    D(fake): {metrics['fake_score']:.4f}")

        # Check metrics are reasonable
        if metrics['loss_d'] < 0 or metrics['loss_g'] < 0:
            print("  ✗ Negative losses detected")
            return False

        if metrics['real_score'] < 0 or metrics['real_score'] > 1:
            print("  ✗ Invalid real_score")
            return False

        print("✓ Training step working\n")
        return True

    except Exception as e:
        print(f"✗ Training step error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostics"""
    print("=" * 60)
    print("DCGAN Backend Diagnostic Tool")
    print("=" * 60)
    print()

    results = []

    # Run all tests
    results.append(("Package imports", check_imports()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("Model creation", test_models()))
    results.append(("Trainer initialization", test_trainer()))
    results.append(("Training step", test_training_step()))
    results.append(("Dataloader", test_dataloader()))

    # Summary
    print("=" * 60)
    print("Diagnostic Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    print()
    if all_passed:
        print("✓ All diagnostics passed!")
        print("\nYour backend is ready. Start with: python main.py")
    else:
        print("✗ Some diagnostics failed")
        print("\nPlease fix the errors above before running the server")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check Python version: python --version (need 3.8+)")
        print("  - Update PyTorch: pip install --upgrade torch torchvision")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
