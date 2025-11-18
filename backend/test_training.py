"""
Test if training actually runs - simpler version to debug
"""

import torch
from trainer import DCGANTrainer
import time

def test_basic_training():
    """Test basic training without FastAPI"""
    print("=" * 60)
    print("Testing DCGAN Training Loop")
    print("=" * 60)

    device = torch.device("cpu")
    trainer = DCGANTrainer(device=device)

    print("\n1. Creating dataloader...")
    print("   This will download MNIST if not present (~10MB)")
    print("   Please wait...")

    try:
        dataloader = trainer.get_dataloader(
            dataset_name='mnist',
            batch_size=64,
            num_workers=0
        )
        print(f"   ✓ Dataloader created: {len(dataloader)} batches")
        print(f"   Dataset size: {len(dataloader.dataset)} images")
    except Exception as e:
        print(f"   ✗ Error creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n2. Testing one training batch...")
    try:
        # Get first batch
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)
        print(f"   Batch shape: {real_images.shape}")

        # Train one step
        print("   Running train_step()...")
        start_time = time.time()
        metrics = trainer.train_step(real_images)
        elapsed = time.time() - start_time

        print(f"   ✓ Training step completed in {elapsed:.2f} seconds")
        print(f"   D loss: {metrics['loss_d']:.4f}")
        print(f"   G loss: {metrics['loss_g']:.4f}")
        print(f"   D(real): {metrics['real_score']:.4f}")
        print(f"   D(fake): {metrics['fake_score']:.4f}")

    except Exception as e:
        print(f"   ✗ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n3. Testing full epoch...")
    try:
        trainer.current_epoch = 0
        batch_count = 0
        epoch_start = time.time()

        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            metrics = trainer.train_step(real_images)
            batch_count += 1

            if i == 0:
                print(f"   Batch 0 completed ({time.time() - epoch_start:.2f}s)")
            elif i == 10:
                print(f"   Batch 10 completed ({time.time() - epoch_start:.2f}s)")
                print("   Stopping test epoch early (enough to verify it works)")
                break

        epoch_time = time.time() - epoch_start
        print(f"   ✓ Processed {batch_count} batches in {epoch_time:.2f} seconds")
        print(f"   Estimated time for full epoch: {epoch_time * len(dataloader) / batch_count:.1f} seconds")

    except Exception as e:
        print(f"   ✗ Error in epoch loop: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n4. Testing image generation...")
    try:
        start_time = time.time()
        fake_images = trainer.generate_images(num_images=4)
        elapsed = time.time() - start_time

        print(f"   ✓ Generated {fake_images.shape[0]} images in {elapsed:.2f} seconds")
        print(f"   Image shape: {fake_images.shape}")

    except Exception as e:
        print(f"   ✗ Error generating images: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n5. Testing base64 conversion...")
    try:
        start_time = time.time()
        img_b64 = trainer.images_to_base64(fake_images, nrow=2)
        elapsed = time.time() - start_time

        print(f"   ✓ Converted to base64 in {elapsed:.2f} seconds")
        print(f"   Base64 length: {len(img_b64)} characters")

    except Exception as e:
        print(f"   ✗ Error in base64 conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TRAINING TESTS PASSED!")
    print("=" * 60)
    print("\nYour training components work correctly.")
    print("If the web app still doesn't work, the issue is in:")
    print("  - FastAPI async handling")
    print("  - WebSocket communication")
    print("  - Frontend React state management")
    print("\nNext: Check browser console for errors (F12)")

    return True

if __name__ == "__main__":
    test_basic_training()
