"""
SUPER SIMPLE SERVER - Minimal version for debugging
If this doesn't work, nothing will!
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_state = {"is_training": False, "current_epoch": 0}
connections = []


@app.get("/")
def root():
    return {"message": "Simple Test Server", "status": "running"}


@app.get("/status")
def get_status():
    print(f"📊 Status requested: {training_state}")
    return training_state


@app.post("/start_training")
async def start_training():
    print("\n" + "="*60)
    print("🚀 START TRAINING CLICKED")
    print("="*60)

    if training_state["is_training"]:
        print("⚠️  Already training")
        return {"success": False, "message": "Already training"}

    # Start background task
    asyncio.create_task(fake_training())

    print("✓ Training task created\n")
    return {"success": True, "message": "Training started"}


@app.post("/generate")
async def generate():
    print("\n📸 GENERATE CLICKED")

    # Return a simple 1x1 red pixel as base64 PNG
    # This is the SIMPLEST possible image
    red_pixel_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    print("✓ Returning red pixel image\n")
    return {
        "success": True,
        "image": red_pixel_png_base64,
        "num_images": 1
    }


async def fake_training():
    """Simulate 3 epochs of training"""
    global training_state

    print("🎬 FAKE TRAINING STARTED")
    training_state["is_training"] = True
    training_state["current_epoch"] = 0

    try:
        for epoch in range(3):
            print(f"\n📚 EPOCH {epoch + 1}/3 STARTING")

            training_state["current_epoch"] = epoch

            # Simulate batches
            for batch in range(5):
                await asyncio.sleep(1)  # 1 second per batch

                print(f"  Batch {batch}/5")

                # Send WebSocket update
                message = {
                    'type': 'batch_update',
                    'epoch': epoch,
                    'batch': batch,
                    'total_batches': 5,
                    'metrics': {
                        'loss_d': 0.5 + random.random() * 0.2,
                        'loss_g': 1.5 + random.random() * 0.5,
                        'real_score': 0.8,
                        'fake_score': 0.3
                    }
                }
                await send_to_all(message)

            # End of epoch
            print(f"✅ EPOCH {epoch + 1} COMPLETE")

            training_state["current_epoch"] = epoch + 1

            message = {
                'type': 'epoch_complete',
                'epoch': epoch,
                'metrics': {
                    'd_loss': 0.6,
                    'g_loss': 1.8,
                    'real_score': 0.8,
                    'fake_score': 0.3
                },
                'sample_image': None,  # Skip images for now
                'all_metrics': {
                    'epoch': epoch,
                    'g_losses': [2.0 - i*0.2 for i in range(epoch+1)],
                    'd_losses': [0.8 - i*0.05 for i in range(epoch+1)],
                    'real_scores': [0.75 + i*0.01 for i in range(epoch+1)],
                    'fake_scores': [0.25 + i*0.01 for i in range(epoch+1)]
                }
            }
            await send_to_all(message)

        print("\n🎉 FAKE TRAINING COMPLETE\n")

    finally:
        training_state["is_training"] = False
        await send_to_all({
            'type': 'training_complete',
            'message': 'Training done'
        })


async def send_to_all(message):
    """Send message to all WebSocket connections"""
    print(f"  📡 Sending {message['type']} to {len(connections)} clients")

    if not connections:
        print("  ⚠️  No clients connected!")
        return

    for ws in connections[:]:  # Copy list to avoid modification during iteration
        try:
            await ws.send_json(message)
            print(f"     ✓ Sent")
        except:
            print(f"     ✗ Failed, removing client")
            try:
                connections.remove(ws)
            except:
                pass


@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)

    print(f"\n🔌 Client connected! Total: {len(connections)}")

    try:
        # Send welcome
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to Simple Test Server',
            'status': training_state
        })
        print("  ✓ Welcome message sent")

        # Keep alive
        while True:
            await asyncio.sleep(1)

    except:
        print("🔌 Client disconnected")
    finally:
        if websocket in connections:
            connections.remove(websocket)
        print(f"  Remaining clients: {len(connections)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🧪 SUPER SIMPLE TEST SERVER")
    print("="*60)
    print("- No real training")
    print("- No complex code")
    print("- Just tests WebSocket communication")
    print("="*60)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
