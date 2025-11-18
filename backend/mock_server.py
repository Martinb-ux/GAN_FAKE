"""
MOCK SERVER - Simulates training without actual GAN
Use this to test if WebSocket and UI work correctly
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import random
import base64
import io
import numpy as np

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    print("⚠️  Pillow not installed - will generate simple noise images")
    HAS_PIL = False

app = FastAPI(title="DCGAN Mock Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections = []
is_training = False
current_epoch = 0
training_task = None


class TrainingConfig(BaseModel):
    dataset: str = "mnist"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0002


class GenerateRequest(BaseModel):
    num_images: int = 16


def create_mock_image_numpy(size=(64, 64)):
    """Create mock image using only numpy (no PIL needed)"""
    # Create random colorful image
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)

    # Add some patterns to make it interesting
    center_x, center_y = size[0] // 2, size[1] // 2
    for i in range(size[0]):
        for j in range(size[1]):
            dist = ((i - center_x)**2 + (j - center_y)**2) ** 0.5
            if dist < size[0] // 3:
                img_array[i, j] = [
                    int(255 * (1 - dist / (size[0] // 3))),
                    random.randint(100, 200),
                    random.randint(100, 200)
                ]

    return img_array


def images_to_base64(num_images=16):
    """Create grid of mock images and encode as base64"""
    if HAS_PIL:
        return images_to_base64_pil(num_images)
    else:
        return images_to_base64_numpy(num_images)


def images_to_base64_numpy(num_images=16):
    """Create image grid using only numpy"""
    nrow = int(num_images ** 0.5)
    img_size = 64
    padding = 2

    # Create grid
    grid_height = nrow * img_size + (nrow + 1) * padding
    grid_width = nrow * img_size + (nrow + 1) * padding
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Place images
    for idx in range(num_images):
        row = idx // nrow
        col = idx % nrow

        y_start = row * img_size + (row + 1) * padding
        x_start = col * img_size + (col + 1) * padding

        img = create_mock_image_numpy((img_size, img_size))
        grid[y_start:y_start+img_size, x_start:x_start+img_size] = img

    # Convert to PNG bytes
    try:
        from PIL import Image
        img_pil = Image.fromarray(grid, 'RGB')
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
    except:
        # Fallback: create a simple PNG manually (won't work well but won't crash)
        # For now, just return a placeholder
        print("⚠️  Cannot create image without Pillow")
        return ""

    img_str = base64.b64encode(img_bytes).decode()
    return img_str


def images_to_base64_pil(num_images=16):
    """Create grid of mock images using PIL"""
    nrow = int(num_images ** 0.5)
    ncol = nrow
    img_size = 64
    padding = 2

    # Create individual images
    images = []
    for i in range(num_images):
        img = Image.new('RGB', (img_size, img_size), color=(
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200)
        ))
        draw = ImageDraw.Draw(img)

        # Draw random shapes
        for _ in range(5):
            x1, y1 = random.randint(0, img_size), random.randint(0, img_size)
            x2, y2 = random.randint(0, img_size), random.randint(0, img_size)
            draw.rectangle([x1, y1, x2, y2], fill=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))

        images.append(img)

    # Create grid
    grid_width = ncol * img_size + (ncol + 1) * padding
    grid_height = nrow * img_size + (nrow + 1) * padding
    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

    # Paste images
    for idx, img in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        x = col * img_size + (col + 1) * padding
        y = row * img_size + (row + 1) * padding
        grid.paste(img, (x, y))

    # Convert to base64
    buffered = io.BytesIO()
    grid.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


@app.get("/")
async def root():
    return {
        "message": "DCGAN MOCK SERVER - For Testing Only",
        "warning": "This simulates training, does not actually train a GAN",
        "purpose": "Test if WebSocket and UI communication works"
    }


@app.get("/status")
async def get_status():
    return {
        "is_training": is_training,
        "current_epoch": current_epoch,
        "device": "mock"
    }


@app.get("/metrics")
async def get_metrics():
    # Generate fake but realistic-looking metrics
    num_epochs = current_epoch + 1
    return {
        "epoch": current_epoch,
        "g_losses": [2.5 - i * 0.1 for i in range(num_epochs)],
        "d_losses": [0.8 - i * 0.02 for i in range(num_epochs)],
        "real_scores": [0.7 + i * 0.02 for i in range(num_epochs)],
        "fake_scores": [0.2 + i * 0.02 for i in range(num_epochs)]
    }


@app.post("/generate")
async def generate_images(request: GenerateRequest):
    print(f"\n📸 Generate request received: {request.num_images} images")
    print("  Simulating image generation...")

    try:
        await asyncio.sleep(0.5)  # Simulate processing time

        img_b64 = images_to_base64(request.num_images)

        if not img_b64:
            print("  ⚠️  Image generation returned empty string")
            return {
                "success": False,
                "error": "Pillow not installed - cannot generate images"
            }

        print(f"  ✓ Generated {request.num_images} mock images, base64 length: {len(img_b64)}")
        return {
            "success": True,
            "image": img_b64,
            "num_images": request.num_images
        }
    except Exception as e:
        print(f"  ✗ Error generating images: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/start_training")
async def start_training(config: TrainingConfig):
    global training_task, is_training

    print("\n" + "="*60)
    print("🚀 START TRAINING REQUEST RECEIVED (MOCK MODE)")
    print(f"Config: {config}")
    print("="*60)

    if is_training:
        print("⚠️  Training already in progress")
        return {"success": False, "message": "Training already in progress"}

    # Start mock training
    training_task = asyncio.create_task(mock_training(config.epochs))

    print("✓ Mock training task created\n")
    return {
        "success": True,
        "message": "Mock training started",
        "config": config.dict()
    }


@app.post("/stop_training")
async def stop_training():
    global is_training, training_task

    print("🛑 Stop training requested")

    if not is_training:
        return {"success": False, "message": "No training in progress"}

    is_training = False

    if training_task:
        training_task.cancel()
        try:
            await training_task
        except asyncio.CancelledError:
            pass

    print("✓ Training stopped\n")
    return {"success": True, "message": "Training stopped"}


async def mock_training(num_epochs):
    """Simulate training with realistic timing and updates"""
    global is_training, current_epoch

    is_training = True
    print("🎬 Starting mock training simulation...")
    print(f"Simulating {num_epochs} epochs\n")

    try:
        for epoch in range(num_epochs):
            if not is_training:
                print("⏹️  Training stopped by user")
                break

            current_epoch = epoch
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Simulate batches (16 batches for 1000 samples / 64 batch size)
            num_batches = 16

            for batch in range(num_batches):
                if not is_training:
                    break

                # Simulate processing time
                await asyncio.sleep(0.5)  # Half second per batch

                # Generate realistic metrics
                d_loss = 0.8 - (epoch * 0.05) + random.uniform(-0.1, 0.1)
                g_loss = 2.5 - (epoch * 0.15) + random.uniform(-0.2, 0.2)
                real_score = 0.7 + (epoch * 0.02) + random.uniform(-0.05, 0.05)
                fake_score = 0.2 + (epoch * 0.03) + random.uniform(-0.05, 0.05)

                step_metrics = {
                    'loss_d': max(0.1, d_loss),
                    'loss_g': max(0.5, g_loss),
                    'real_score': min(0.95, max(0.5, real_score)),
                    'fake_score': min(0.6, max(0.1, fake_score))
                }

                # Send batch update every 5 batches
                if batch % 5 == 0:
                    print(f"  📊 Batch {batch}/{num_batches} - "
                          f"D_loss={step_metrics['loss_d']:.4f}, "
                          f"G_loss={step_metrics['loss_g']:.4f}")

                    await broadcast_update({
                        'type': 'batch_update',
                        'epoch': epoch,
                        'batch': batch,
                        'total_batches': num_batches,
                        'metrics': step_metrics
                    })

            # End of epoch
            print(f"\n✓ Epoch {epoch + 1} completed!")
            print(f"  Generating sample images...")

            # Generate mock sample image
            sample_image = images_to_base64(16)

            # Average metrics for epoch
            epoch_metrics = {
                'd_loss': 0.8 - (epoch * 0.05),
                'g_loss': 2.5 - (epoch * 0.15),
                'real_score': 0.7 + (epoch * 0.02),
                'fake_score': 0.2 + (epoch * 0.03)
            }

            all_metrics = {
                'epoch': epoch,
                'g_losses': [2.5 - i * 0.15 for i in range(epoch + 1)],
                'd_losses': [0.8 - i * 0.05 for i in range(epoch + 1)],
                'real_scores': [0.7 + i * 0.02 for i in range(epoch + 1)],
                'fake_scores': [0.2 + i * 0.03 for i in range(epoch + 1)]
            }

            print(f"  📤 Sending epoch complete update via WebSocket")
            await broadcast_update({
                'type': 'epoch_complete',
                'epoch': epoch,
                'metrics': epoch_metrics,
                'sample_image': sample_image,
                'all_metrics': all_metrics
            })

            print(f"  D_loss={epoch_metrics['d_loss']:.4f}, "
                  f"G_loss={epoch_metrics['g_loss']:.4f}, "
                  f"D(real)={epoch_metrics['real_score']:.4f}, "
                  f"D(fake)={epoch_metrics['fake_score']:.4f}")

        print(f"\n{'='*60}")
        print("🎉 MOCK TRAINING COMPLETE!")
        print(f"{'='*60}\n")

    except asyncio.CancelledError:
        print("⏹️  Training cancelled")
    except Exception as e:
        print(f"❌ Error in mock training: {e}")
    finally:
        is_training = False
        await broadcast_update({
            'type': 'training_complete',
            'message': 'Mock training completed'
        })


async def broadcast_update(message):
    """Broadcast to all WebSocket clients"""
    if not active_connections:
        print(f"  ⚠️  No WebSocket clients - message type: {message.get('type')}")
        return

    print(f"  📡 Broadcasting to {len(active_connections)} client(s): {message.get('type')}")

    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
            print(f"     ✓ Sent to client")
        except Exception as e:
            print(f"     ✗ Failed to send: {e}")
            disconnected.append(connection)

    for conn in disconnected:
        try:
            active_connections.remove(conn)
        except ValueError:
            pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)

    print(f"\n🔌 WebSocket client connected! Total clients: {len(active_connections)}")

    try:
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to MOCK server (simulated training)',
            'status': {
                'is_training': is_training,
                'current_epoch': current_epoch
            }
        })
        print("  ✓ Sent welcome message to client")

        # Keep alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                await websocket.send_json({'type': 'pong'})
            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"  Remaining clients: {len(active_connections)}")


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🧪 DCGAN MOCK SERVER")
    print("="*60)
    print("This is a TEST server that SIMULATES training")
    print("No actual GAN training happens")
    print("Use this to verify WebSocket and UI work correctly")
    print("="*60)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
