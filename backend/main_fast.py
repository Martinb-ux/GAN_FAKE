"""
FastAPI Server for DCGAN Demo - FAST MODE
Uses smaller dataset (1000 samples) for quick testing
~15 seconds per epoch instead of ~10 minutes
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import asyncio
import json
from trainer import DCGANTrainer
from torch.utils.data import Subset
import random
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DCGAN Demo API - Fast Mode")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trainer instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = DCGANTrainer(device=device)
training_task = None
active_connections = []


class TrainingConfig(BaseModel):
    dataset: str = "mnist"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0002


class GenerateRequest(BaseModel):
    num_images: int = 16


@app.get("/")
async def root():
    """API root"""
    return {
        "message": "DCGAN Demo API - FAST MODE (1000 samples)",
        "mode": "demo",
        "samples_per_dataset": 1000,
        "estimated_time_per_epoch": "15-20 seconds",
        "endpoints": {
            "/start_training": "POST - Start GAN training",
            "/stop_training": "POST - Stop GAN training",
            "/status": "GET - Get training status",
            "/generate": "POST - Generate synthetic images",
            "/metrics": "GET - Get training metrics",
            "/ws": "WebSocket - Real-time training updates"
        }
    }


@app.get("/status")
async def get_status():
    """Get current training status"""
    return {
        "is_training": trainer.is_training,
        "current_epoch": trainer.current_epoch,
        "device": str(device),
        "mode": "fast"
    }


@app.get("/metrics")
async def get_metrics():
    """Get training metrics"""
    return trainer.get_metrics()


@app.post("/generate")
async def generate_images(request: GenerateRequest):
    """Generate synthetic images"""
    try:
        num_images = min(request.num_images, 64)
        logger.info(f"Generating {num_images} images...")
        fake_images = trainer.generate_images(num_images=num_images)
        image_b64 = trainer.images_to_base64(fake_images, nrow=int(num_images**0.5))
        logger.info("Images generated successfully")

        return {
            "success": True,
            "image": image_b64,
            "num_images": num_images
        }
    except Exception as e:
        logger.error(f"Error generating images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start_training")
async def start_training(config: TrainingConfig):
    """Start GAN training with small dataset"""
    global training_task

    logger.info("="*60)
    logger.info("START TRAINING REQUEST RECEIVED")
    logger.info(f"Config: {config}")
    logger.info("="*60)

    if trainer.is_training:
        logger.warning("Training already in progress!")
        return {"success": False, "message": "Training already in progress"}

    try:
        logger.info("Step 1: Creating FAST dataloader with 1000 samples...")

        # Get full dataloader
        logger.info(f"  Loading {config.dataset} dataset...")
        full_dataloader = trainer.get_dataloader(
            dataset_name=config.dataset,
            batch_size=config.batch_size
        )
        logger.info(f"  ✓ Full dataset loaded: {len(full_dataloader.dataset)} images")

        # Create small subset (1000 samples for fast training)
        logger.info("Step 2: Creating subset...")
        dataset = full_dataloader.dataset
        num_samples = min(1000, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)
        subset = Subset(dataset, indices)
        logger.info(f"  ✓ Subset created: {num_samples} samples")

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )

        logger.info(f"Step 3: Dataloader ready - {len(dataloader)} batches")

        # Start training in background
        logger.info("Step 4: Starting async training task...")
        training_task = asyncio.create_task(
            run_training(trainer, dataloader, config.epochs)
        )
        logger.info("  ✓ Training task created and started!")

        return {
            "success": True,
            "message": f"Fast training started ({num_samples} samples, {len(dataloader)} batches)",
            "config": config.dict()
        }
    except Exception as e:
        logger.error(f"✗ Error starting training: {e}", exc_info=True)
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


async def run_training(trainer: DCGANTrainer, dataloader, num_epochs):
    """Run training loop asynchronously"""
    trainer.is_training = True
    logger.info("="*60)
    logger.info("RUN_TRAINING FUNCTION CALLED")
    logger.info(f"Starting FAST training for {num_epochs} epochs on {trainer.device}")
    logger.info(f"Dataset size: {len(dataloader.dataset)}, Batches: {len(dataloader)}")
    logger.info("="*60)

    try:
        for epoch in range(num_epochs):
            logger.info(f"\n>>> STARTING EPOCH {epoch+1}/{num_epochs}")

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
            logger.info(f"Beginning batch iteration ({len(dataloader)} total batches)...")

            for i, data in enumerate(dataloader):
                try:
                    if i == 0:
                        logger.info(f"  Processing batch 0...")

                    real_images = data[0].to(trainer.device)

                    # Train step
                    step_metrics = trainer.train_step(real_images)

                    if i == 0:
                        logger.info(f"  ✓ Batch 0 complete! D_loss={step_metrics['loss_d']:.4f}, G_loss={step_metrics['loss_g']:.4f}")

                    # Accumulate metrics
                    epoch_metrics['g_loss'] += step_metrics['loss_g']
                    epoch_metrics['d_loss'] += step_metrics['loss_d']
                    epoch_metrics['real_score'] += step_metrics['real_score']
                    epoch_metrics['fake_score'] += step_metrics['fake_score']
                    batch_count += 1

                    # Send updates every 5 batches (more frequent than normal)
                    if i % 5 == 0:
                        logger.info(f"  Batch {i}/{len(dataloader)} - Sending WebSocket update")
                        await broadcast_update({
                            'type': 'batch_update',
                            'epoch': epoch,
                            'batch': i,
                            'total_batches': len(dataloader),
                            'metrics': step_metrics
                        })

                    # Allow other async tasks to run more frequently
                    if i % 2 == 0:
                        await asyncio.sleep(0)

                except Exception as e:
                    logger.error(f"✗ Error in batch {i}: {e}", exc_info=True)
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
                f"✓ Epoch [{epoch+1}/{num_epochs}] completed - "
                f"Loss_D: {epoch_metrics['d_loss']:.4f} "
                f"Loss_G: {epoch_metrics['g_loss']:.4f} "
                f"D(x): {epoch_metrics['real_score']:.4f} "
                f"D(G(z)): {epoch_metrics['fake_score']:.4f}"
            )

            await asyncio.sleep(0.1)

        logger.info("Training completed successfully!")

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
        logger.warning(f"No WebSocket clients connected - message type: {message.get('type')}")
        return

    logger.debug(f"Broadcasting to {len(active_connections)} clients: {message.get('type')}")

    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            disconnected.append(connection)

    for conn in disconnected:
        try:
            active_connections.remove(conn)
        except ValueError:
            pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(active_connections)}")

    try:
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to DCGAN training server (FAST MODE - 1000 samples)',
            'status': {
                'is_training': trainer.is_training,
                'current_epoch': trainer.current_epoch
            }
        })

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                await websocket.send_json({'type': 'pong', 'data': data})
            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket client removed. Total clients: {len(active_connections)}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("DCGAN Backend - FAST MODE")
    print("=" * 60)
    print("Dataset size: 1000 samples (instead of 60,000)")
    print("Expected time per epoch: 15-20 seconds")
    print("This is for TESTING - results will be lower quality")
    print("=" * 60)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
