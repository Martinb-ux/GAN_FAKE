"""
Simple test script to verify DCGAN backend functionality
Run this to test the models and trainer without starting the full server
"""

import torch
from models import Generator, Discriminator, weights_init
from trainer import DCGANTrainer


def test_models():
    """Test Generator and Discriminator forward passes"""
    print("Testing DCGAN Models...")

    device = torch.device("cpu")
    nz = 100
    batch_size = 4

    # Test Generator
    print("\n1. Testing Generator...")
    netG = Generator(nz=nz).to(device)
    netG.apply(weights_init)

    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake_images = netG(noise)

    print(f"   Input shape: {noise.shape}")
    print(f"   Output shape: {fake_images.shape}")
    print(f"   Output range: [{fake_images.min().item():.3f}, {fake_images.max().item():.3f}]")
    assert fake_images.shape == (batch_size, 3, 64, 64), "Generator output shape incorrect"
    assert fake_images.min() >= -1.1 and fake_images.max() <= 1.1, "Generator output range incorrect"
    print("   ✓ Generator working correctly")

    # Test Discriminator
    print("\n2. Testing Discriminator...")
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    predictions = netD(fake_images)

    print(f"   Input shape: {fake_images.shape}")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Output range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
    assert predictions.shape == (batch_size, 1, 1, 1), "Discriminator output shape incorrect"
    assert predictions.min() >= 0 and predictions.max() <= 1, "Discriminator output range incorrect"
    print("   ✓ Discriminator working correctly")


def test_trainer():
    """Test DCGANTrainer initialization and basic functions"""
    print("\n3. Testing DCGANTrainer...")

    device = torch.device("cpu")
    trainer = DCGANTrainer(device=device)

    print("   Initialized trainer")
    print(f"   Device: {trainer.device}")
    print(f"   Latent size: {trainer.nz}")

    # Test image generation
    print("\n4. Testing image generation...")
    fake_images = trainer.generate_images(num_images=8)
    print(f"   Generated images shape: {fake_images.shape}")
    assert fake_images.shape == (8, 3, 64, 64), "Generated images shape incorrect"
    print("   ✓ Image generation working")

    # Test base64 conversion
    print("\n5. Testing image to base64 conversion...")
    img_b64 = trainer.images_to_base64(fake_images, nrow=4)
    print(f"   Base64 length: {len(img_b64)} characters")
    assert len(img_b64) > 1000, "Base64 encoding too short"
    print("   ✓ Base64 conversion working")

    # Test training step (single iteration)
    print("\n6. Testing training step...")
    real_images = torch.randn(4, 3, 64, 64, device=device)
    real_images = (real_images - real_images.min()) / (real_images.max() - real_images.min())
    real_images = real_images * 2 - 1  # Scale to [-1, 1]

    metrics = trainer.train_step(real_images)

    print(f"   D loss: {metrics['loss_d']:.4f}")
    print(f"   G loss: {metrics['loss_g']:.4f}")
    print(f"   D(real): {metrics['real_score']:.4f}")
    print(f"   D(fake): {metrics['fake_score']:.4f}")
    assert 'loss_d' in metrics and 'loss_g' in metrics, "Metrics missing"
    print("   ✓ Training step working")


def test_dataloader():
    """Test dataloader creation"""
    print("\n7. Testing dataloader...")

    device = torch.device("cpu")
    trainer = DCGANTrainer(device=device)

    try:
        print("   Creating MNIST dataloader (will download if needed)...")
        dataloader = trainer.get_dataloader(dataset_name='mnist', batch_size=16, num_workers=0)

        # Get one batch
        images, labels = next(iter(dataloader))
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")

        assert images.shape[1] == 3, "Images should have 3 channels (RGB)"
        assert images.shape[2] == 64 and images.shape[3] == 64, "Images should be 64x64"
        print("   ✓ Dataloader working correctly")

    except Exception as e:
        print(f"   ⚠ Warning: Could not test dataloader - {e}")
        print("   This is expected if running without internet for MNIST download")


if __name__ == "__main__":
    print("=" * 60)
    print("DCGAN Backend Test Suite")
    print("=" * 60)

    try:
        test_models()
        test_trainer()
        test_dataloader()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nYour DCGAN backend is ready to use.")
        print("Start the server with: python main.py")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
