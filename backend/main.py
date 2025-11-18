"""
FastAPI Server for DCGAN Demo
Provides REST API and WebSocket for real-time training visualization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import asyncio
import json
from trainer import DCGANTrainer
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DCGAN Demo API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trainer instance
# Support Apple Silicon (MPS), CUDA, and CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

trainer = DCGANTrainer(device=device)
training_task = None
active_connections = []


class TrainingConfig(BaseModel):
    dataset: str = "mnist"  # mnist or fashion_mnist
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0002
    device: str = "mps"  # mps, cuda, or cpu


class GenerateRequest(BaseModel):
    num_images: int = 16


@app.get("/")
async def root():
    """API root"""
    return {
        "message": "DCGAN Demo API",
        "endpoints": {
            "/start_training": "POST - Start GAN training",
            "/stop_training": "POST - Stop GAN training",
            "/status": "GET - Get training status",
            "/generate": "POST - Generate synthetic images",
            "/metrics": "GET - Get training metrics",
            "/save_model": "POST - Save model checkpoint",
            "/load_model": "POST - Load model checkpoint",
            "/ws": "WebSocket - Real-time training updates"
        }
    }


@app.get("/status")
async def get_status():
    """Get current training status"""
    return {
        "is_training": trainer.is_training,
        "current_epoch": trainer.current_epoch,
        "device": str(device)
    }


@app.get("/metrics")
async def get_metrics():
    """Get training metrics"""
    return trainer.get_metrics()


@app.post("/generate")
async def generate_images(request: GenerateRequest):
    """
    Generate synthetic images

    Args:
        request: GenerateRequest with num_images

    Returns:
        Base64 encoded image grid
    """
    try:
        num_images = min(request.num_images, 64)  # Limit to 64 images
        fake_images = trainer.generate_images(num_images=num_images)
        image_b64 = trainer.images_to_base64(fake_images, nrow=int(num_images**0.5))

        return {
            "success": True,
            "image": image_b64,
            "num_images": num_images
        }
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start_training")
async def start_training(config: TrainingConfig):
    """
    Start GAN training

    Args:
        config: TrainingConfig with training parameters

    Returns:
        Training start confirmation
    """
    global training_task, trainer, device

    if trainer.is_training:
        return {"success": False, "message": "Training already in progress"}

    try:
        # Switch device if needed
        requested_device = torch.device(config.device)
        if str(requested_device) != str(trainer.device):
            logger.info(f"Switching device from {trainer.device} to {requested_device}")
            device = requested_device
            trainer = DCGANTrainer(device=device)
            logger.info(f"Trainer recreated with device: {device}")

        # Create dataloader
        dataloader = trainer.get_dataloader(
            dataset_name=config.dataset,
            batch_size=config.batch_size
        )

        # Start training in background
        training_task = asyncio.create_task(
            run_training(trainer, dataloader, config.epochs)
        )

        return {
            "success": True,
            "message": f"Training started on {config.device}",
            "config": config.dict()
        }
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop_training")
async def stop_training():
    """Stop ongoing training"""
    global training_task

    if not trainer.is_training:
        return {"success": False, "message": "No training in progress"}

    trainer.is_training = False

    if training_task:
        training_task.cancel()
        try:
            await training_task
        except asyncio.CancelledError:
            pass

    return {"success": True, "message": "Training stopped"}


@app.post("/save_model")
async def save_model():
    """
    Save the current model checkpoint

    Returns:
        Save confirmation with file path
    """
    if trainer.is_training:
        return {"success": False, "message": "Cannot save model during training"}

    try:
        checkpoint_path = "dcgan_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

        return {
            "success": True,
            "message": "Model saved successfully",
            "path": checkpoint_path
        }
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model():
    """
    Load a model checkpoint

    Returns:
        Load confirmation with file path
    """
    if trainer.is_training:
        return {"success": False, "message": "Cannot load model during training"}

    try:
        checkpoint_path = "dcgan_checkpoint.pth"
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Model loaded from {checkpoint_path}")

        return {
            "success": True,
            "message": "Model loaded successfully",
            "path": checkpoint_path
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No saved model found")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training(trainer: DCGANTrainer, dataloader, num_epochs):
    """
    Run training loop asynchronously

    Args:
        trainer: DCGANTrainer instance
        dataloader: PyTorch DataLoader
        num_epochs: Number of epochs to train
    """
    trainer.is_training = True
    logger.info(f"Starting training for {num_epochs} epochs on {trainer.device}")

    try:
        for epoch in range(num_epochs):
            if not trainer.is_training:
                logger.info("Training stopped by user")
                break

            trainer.current_epoch = epoch
            epoch_metrics = {
                'g_loss': 0,
                'd_loss': 0,
                'real_score': 0,
                'fake_score': 0
            }

            batch_count = 0
            for i, data in enumerate(dataloader):
                try:
                    real_images = data[0].to(trainer.device)

                    # Train step
                    step_metrics = trainer.train_step(real_images)

                    # Accumulate metrics
                    epoch_metrics['g_loss'] += step_metrics['loss_g']
                    epoch_metrics['d_loss'] += step_metrics['loss_d']
                    epoch_metrics['real_score'] += step_metrics['real_score']
                    epoch_metrics['fake_score'] += step_metrics['fake_score']
                    batch_count += 1

                    # Send updates every 5 batches (even more frequent for GPU)
                    if i % 5 == 0:
                        await broadcast_update({
                            'type': 'batch_update',
                            'epoch': epoch,
                            'batch': i,
                            'total_batches': len(dataloader),
                            'metrics': step_metrics
                        })
                        if i % 50 == 0:  # Only log every 50 to avoid spam
                            logger.info(f"Epoch {epoch}, Batch {i}/{len(dataloader)}")

                    # CRITICAL: Yield to event loop after EVERY batch to keep WebSocket alive
                    # This is especially important with fast GPU training
                    await asyncio.sleep(0)

                except Exception as e:
                    logger.error(f"Error in batch {i}: {e}")
                    continue

            # Average metrics over epoch
            if batch_count > 0:
                epoch_metrics = {k: v / batch_count for k, v in epoch_metrics.items()}
            else:
                logger.error("No batches processed in epoch")
                continue

            # Store metrics
            trainer.metrics['g_losses'].append(epoch_metrics['g_loss'])
            trainer.metrics['d_losses'].append(epoch_metrics['d_loss'])
            trainer.metrics['real_scores'].append(epoch_metrics['real_score'])
            trainer.metrics['fake_scores'].append(epoch_metrics['fake_score'])

            # Generate sample images
            try:
                fake_images = trainer.generate_images(num_images=16, noise=trainer.fixed_noise[:16])
                image_b64 = trainer.images_to_base64(fake_images, nrow=4)
            except Exception as e:
                logger.error(f"Error generating sample images: {e}")
                image_b64 = None

            # Send epoch update
            await broadcast_update({
                'type': 'epoch_complete',
                'epoch': epoch,
                'metrics': epoch_metrics,
                'sample_image': image_b64,
                'all_metrics': trainer.get_metrics()
            })

            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss_D: {epoch_metrics['d_loss']:.4f} "
                f"Loss_G: {epoch_metrics['g_loss']:.4f} "
                f"D(x): {epoch_metrics['real_score']:.4f} "
                f"D(G(z)): {epoch_metrics['fake_score']:.4f}"
            )

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        logger.info("Training cancelled")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        await broadcast_update({
            'type': 'error',
            'message': f"Training error: {str(e)}"
        })
    finally:
        trainer.is_training = False
        await broadcast_update({
            'type': 'training_complete',
            'message': 'Training completed or stopped'
        })


async def broadcast_update(message):
    """Broadcast message to all connected WebSocket clients"""
    if not active_connections:
        return  # No clients connected, skip

    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            disconnected.append(connection)

    # Remove disconnected clients
    for conn in disconnected:
        try:
            active_connections.remove(conn)
        except ValueError:
            pass  # Already removed


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training updates

    Clients connect here to receive:
    - Batch updates during training
    - Epoch completion updates with sample images
    - Training metrics
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(active_connections)}")

    async def send_heartbeat():
        """Send periodic heartbeat to keep connection alive"""
        while True:
            try:
                await asyncio.sleep(5)  # Send heartbeat every 5 seconds
                if websocket in active_connections:
                    await websocket.send_json({'type': 'heartbeat', 'timestamp': asyncio.get_event_loop().time()})
            except Exception:
                break

    try:
        # Send initial status
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to DCGAN training server',
            'status': {
                'is_training': trainer.is_training,
                'current_epoch': trainer.current_epoch
            }
        })

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(send_heartbeat())

        # Keep connection alive - listen for client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                # Echo back any received messages (for ping/pong)
                await websocket.send_json({'type': 'pong', 'data': data})
            except asyncio.TimeoutError:
                # No message received in 10 seconds - this is fine
                continue

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket client removed. Total clients: {len(active_connections)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
