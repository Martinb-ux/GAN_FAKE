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
import os
import time
from trainer import DCGANTrainer
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DCGAN Demo API")

# CORS middleware for React frontend
# Allow localhost for development and any origin for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS[0] != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trainer instance
# Support Apple Silicon (MPS), CUDA, and CPU
def get_device():
    """Detect available device with proper error handling"""
    # Check for CUDA first
    if torch.cuda.is_available():
        logger.info("Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")

    # Check for MPS (Apple Silicon) - verify it actually works
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test if MPS actually works by creating a small tensor
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
            logger.info("Using Apple Silicon GPU (MPS)")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS available but not working: {e}")

    # Fallback to CPU
    logger.info("Using CPU")
    return torch.device("cpu")

device = get_device()

trainer = DCGANTrainer(device=device)
training_task = None
active_connections = []

# Training session tracking
training_sessions = {}  # Store training history for comparison
current_training_start_time = None
current_training_dataset = None


class TrainingConfig(BaseModel):
    dataset: str = "mnist"  # mnist or fashion_mnist
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0002
    device: str = "mps"  # mps, cuda, or cpu


class GenerateRequest(BaseModel):
    num_images: int = 16


class SaveModelRequest(BaseModel):
    model_name: str = "default"  # mnist, fashion_mnist, or default


class LoadModelRequest(BaseModel):
    model_name: str = "default"  # mnist, fashion_mnist, or default


class ListModelsResponse(BaseModel):
    models: list = []


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
            "/save_model": "POST - Save model checkpoint (accepts model_name)",
            "/load_model": "POST - Load model checkpoint (accepts model_name)",
            "/list_models": "GET - List available saved models",
            "/training_sessions": "GET - Get training session history for comparison",
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


class GenerateFromModelRequest(BaseModel):
    model_name: str = "default"  # mnist, fashion_mnist, or default
    num_images: int = 16


@app.post("/generate_from_model")
async def generate_from_model(request: GenerateFromModelRequest):
    """
    Generate images from a specific saved model

    Args:
        request: GenerateFromModelRequest with model_name and num_images

    Returns:
        Base64 encoded image grid from the specified model
    """
    if trainer.is_training:
        return {"success": False, "message": "Cannot generate while training"}

    try:
        # Determine checkpoint path
        if request.model_name == "mnist":
            checkpoint_path = "mnist_checkpoint.pth"
        elif request.model_name == "fashion_mnist":
            checkpoint_path = "fashion_checkpoint.pth"
        else:
            checkpoint_path = "dcgan_checkpoint.pth"

        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")

        # Load the model temporarily
        trainer.load_checkpoint(checkpoint_path)

        # Generate images
        num_images = min(request.num_images, 64)
        fake_images = trainer.generate_images(num_images=num_images)
        image_b64 = trainer.images_to_base64(fake_images, nrow=int(num_images**0.5))

        return {
            "success": True,
            "image": image_b64,
            "num_images": num_images,
            "model_name": request.model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating images from model: {e}")
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
        # Validate and switch device if needed
        requested_device_str = config.device

        # Check if requested device is available
        if requested_device_str == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                requested_device_str = "cpu"
        elif requested_device_str == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                requested_device_str = "cpu"

        requested_device = torch.device(requested_device_str)

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
            run_training(trainer, dataloader, config.epochs, config.dataset)
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
async def save_model(request: SaveModelRequest = None):
    """
    Save the current model checkpoint

    Args:
        request: SaveModelRequest with model_name (mnist, fashion_mnist, or default)

    Returns:
        Save confirmation with file path
    """
    if trainer.is_training:
        return {"success": False, "message": "Cannot save model during training"}

    try:
        # Determine checkpoint filename based on model_name
        model_name = request.model_name if request else "default"
        if model_name == "mnist":
            checkpoint_path = "mnist_checkpoint.pth"
        elif model_name == "fashion_mnist":
            checkpoint_path = "fashion_checkpoint.pth"
        else:
            checkpoint_path = "dcgan_checkpoint.pth"

        trainer.save_checkpoint(checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

        return {
            "success": True,
            "message": f"Model saved successfully as {model_name}",
            "path": checkpoint_path,
            "model_name": model_name
        }
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model(request: LoadModelRequest = None):
    """
    Load a model checkpoint

    Args:
        request: LoadModelRequest with model_name (mnist, fashion_mnist, or default)

    Returns:
        Load confirmation with file path and metrics
    """
    if trainer.is_training:
        return {"success": False, "message": "Cannot load model during training"}

    try:
        # Determine checkpoint filename based on model_name
        model_name = request.model_name if request else "default"
        if model_name == "mnist":
            checkpoint_path = "mnist_checkpoint.pth"
        elif model_name == "fashion_mnist":
            checkpoint_path = "fashion_checkpoint.pth"
        else:
            checkpoint_path = "dcgan_checkpoint.pth"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Model loaded from {checkpoint_path}")

        return {
            "success": True,
            "message": f"Model loaded successfully ({model_name})",
            "path": checkpoint_path,
            "model_name": model_name,
            "metrics": trainer.get_metrics()
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No saved model found for {model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_models")
async def list_models():
    """
    List available saved model checkpoints

    Returns:
        List of available models with their info
    """
    models = []
    model_files = {
        "mnist": "mnist_checkpoint.pth",
        "fashion_mnist": "fashion_checkpoint.pth",
        "default": "dcgan_checkpoint.pth"
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            stat = os.stat(path)
            models.append({
                "name": name,
                "path": path,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": time.ctime(stat.st_mtime)
            })

    return {"models": models}


@app.get("/training_sessions")
async def get_training_sessions():
    """
    Get training session history for comparison

    Returns:
        Dictionary of training sessions with metrics
    """
    return {"sessions": training_sessions}


async def run_training(trainer: DCGANTrainer, dataloader, num_epochs, dataset_name: str = "unknown"):
    """
    Run training loop asynchronously

    Args:
        trainer: DCGANTrainer instance
        dataloader: PyTorch DataLoader
        num_epochs: Number of epochs to train
        dataset_name: Name of the dataset being trained on
    """
    global current_training_start_time, current_training_dataset, training_sessions

    trainer.is_training = True
    current_training_start_time = time.time()
    current_training_dataset = dataset_name
    logger.info(f"Starting training for {num_epochs} epochs on {trainer.device} with {dataset_name}")

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

        # Calculate training duration and save session
        training_duration = time.time() - current_training_start_time if current_training_start_time else 0
        session_data = {
            "dataset": current_training_dataset,
            "epochs": trainer.current_epoch + 1,
            "training_time_seconds": round(training_duration, 2),
            "training_time_formatted": f"{int(training_duration // 60)}m {int(training_duration % 60)}s",
            "device": str(trainer.device),
            "final_g_loss": trainer.metrics['g_losses'][-1] if trainer.metrics['g_losses'] else None,
            "final_d_loss": trainer.metrics['d_losses'][-1] if trainer.metrics['d_losses'] else None,
            "final_real_score": trainer.metrics['real_scores'][-1] if trainer.metrics['real_scores'] else None,
            "final_fake_score": trainer.metrics['fake_scores'][-1] if trainer.metrics['fake_scores'] else None,
            "metrics": trainer.get_metrics(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Store session by dataset name
        training_sessions[current_training_dataset] = session_data
        logger.info(f"Training session saved for {current_training_dataset}: {training_duration:.1f}s")

        await broadcast_update({
            'type': 'training_complete',
            'message': 'Training completed or stopped',
            'session': session_data
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
