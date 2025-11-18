# DCGAN Interactive Demo

An interactive web application demonstrating **Deep Convolutional Generative Adversarial Networks (DCGAN)** for synthetic image generation.

## Features

- **Real-time Training Visualization**: Watch the GAN train with live loss curves and sample images
- **Interactive Controls**: Start/stop training, adjust hyperparameters
- **Device Selection**: GPU (NVIDIA CUDA, Apple Silicon MPS) or CPU training
- **Cross-Platform**: Works on Mac, Windows, and Linux ([Platform Guide](PLATFORM_COMPATIBILITY.md))
- **Model Persistence**: Save and load trained models
- **Image Generation**: Generate synthetic images on demand
- **Multiple Datasets**: MNIST digits or Fashion-MNIST clothing items
- **WebSocket Updates**: Real-time metrics and progress updates
- **Professional UI**: Clean, responsive interface with charts and visualizations

## Architecture

### Backend (FastAPI + PyTorch)
- **Generator Network**: Transforms 100-dim noise vector to 64×64 RGB images
- **Discriminator Network**: Binary classifier for real vs fake images
- **Training Loop**: Alternating optimization with Adam optimizer
- **REST API**: Endpoints for training control and image generation
- **WebSocket**: Real-time training updates

### Frontend (React + Vite)
- **Training Dashboard**: Control panel with hyperparameter settings
- **Live Metrics**: Loss curves and discriminator scores (Recharts)
- **Image Display**: Generated samples and training progress
- **Training Logs**: Real-time log viewer
- **Architecture Info**: Visual explanation of DCGAN structure

## DCGAN Implementation Details

Based on the paper: *Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)*

### Generator Architecture
```
Input: z ∈ ℝ¹⁰⁰ (Gaussian noise)

Layer 1: ConvTranspose2d(100 → 512, 4×4)   → BatchNorm → ReLU
Layer 2: ConvTranspose2d(512 → 256, 8×8)   → BatchNorm → ReLU
Layer 3: ConvTranspose2d(256 → 128, 16×16) → BatchNorm → ReLU
Layer 4: ConvTranspose2d(128 → 64, 32×32)  → BatchNorm → ReLU
Layer 5: ConvTranspose2d(64 → 3, 64×64)    → Tanh

Output: 3×64×64 RGB image, values in [-1, 1]
```

### Discriminator Architecture
```
Input: 3×64×64 RGB image

Layer 1: Conv2d(3 → 64, 32×32)    → LeakyReLU(0.2)
Layer 2: Conv2d(64 → 128, 16×16)  → BatchNorm → LeakyReLU(0.2)
Layer 3: Conv2d(128 → 256, 8×8)   → BatchNorm → LeakyReLU(0.2)
Layer 4: Conv2d(256 → 512, 4×4)   → BatchNorm → LeakyReLU(0.2)
Layer 5: Conv2d(512 → 1, 1×1)     → Sigmoid

Output: Probability ∈ [0, 1]
```

### Training Algorithm
```
For each iteration:
  // Update Discriminator
  1. Sample real images: {x⁽¹⁾, ..., x⁽ᵐ⁾} ~ p_data
  2. Sample noise: {z⁽¹⁾, ..., z⁽ᵐ⁾} ~ N(0, I)
  3. Generate fakes: x̃ = G(z)
  4. Compute D loss: L_D = -log D(x) - log(1 - D(x̃))
  5. Update: θ_D ← θ_D - α ∇_θ_D L_D

  // Update Generator
  6. Sample noise: {z⁽¹⁾, ..., z⁽ᵐ⁾} ~ N(0, I)
  7. Compute G loss: L_G = -log D(G(z))  (non-saturating)
  8. Update: θ_G ← θ_G - α ∇_θ_G L_G
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- pip and npm
- (Optional) GPU for acceleration:
  - **Mac:** Apple Silicon (M1/M2/M3/M4) with MPS support
  - **Windows/Linux:** NVIDIA GPU with CUDA support (see [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md))

### Quick Start

The easiest way to run the application is using the provided startup scripts:

1. **Start Backend** (in one terminal):
```bash
cd dcgan-demo
chmod +x start_backend.sh
./start_backend.sh
```

2. **Start Frontend** (in another terminal):
```bash
cd dcgan-demo
chmod +x start_frontend.sh
./start_frontend.sh
```

3. **Open Browser**: Navigate to `http://localhost:5173`

The startup scripts automatically:
- Create virtual environments
- Install dependencies
- Run diagnostics
- Start the servers

### Manual Setup (Alternative)

#### Backend Setup

1. Navigate to backend directory:
```bash
cd dcgan-demo/backend
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

#### Frontend Setup

1. Navigate to frontend directory:
```bash
cd dcgan-demo/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## Usage

1. **Start Both Servers**:
   - Backend: `./start_backend.sh` or `python main.py` (runs on port 8000)
   - Frontend: `./start_frontend.sh` or `npm run dev` (runs on port 5173)

2. **Open the Web App**: Navigate to `http://localhost:5173`

3. **Configure Training**:
   - Choose dataset (MNIST or Fashion-MNIST)
   - **Select device** (GPU or CPU) - see performance comparison below
   - Set epochs (recommended: 10-20 for demo)
   - Adjust batch size (default: 64)
   - Set learning rate (default: 0.0002)

4. **Start Training**: Click "Start Training" and watch:
   - Real-time loss curves
   - Generated sample images each epoch
   - Discriminator scores (D(real) and D(fake))
   - Training logs

5. **Save/Load Models**:
   - Click "Save Model" to save current weights
   - Click "Load Model" to restore a saved checkpoint
   - Models are saved as `dcgan_checkpoint.pth` in the backend directory

6. **Generate Images**: After training, use "Generate Images" to create synthetic samples

## GPU vs CPU Performance

The application supports both GPU (Apple Silicon MPS) and CPU training. The performance difference is significant:

### Training Performance Comparison

| Device | Time per Epoch | Total Time (10 epochs) | Speed Factor |
|--------|---------------|----------------------|--------------|
| **NVIDIA RTX 3060 (CUDA)** | 30-60 seconds | 5-10 minutes | **30-40x faster** |
| **Apple M4 Max (MPS)** | 1-3 minutes | 10-30 minutes | **20-30x faster** |
| **CPU** | 20-50 minutes | 200-500 minutes | Baseline |

**Note:** NVIDIA GPUs typically provide the best performance. See [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md) for Windows setup.

### Technical Details

#### GPU Acceleration (Apple Silicon MPS)

**How it works:**
- Uses Apple's Metal Performance Shaders (MPS) backend
- PyTorch operations are executed on the GPU cores
- Parallel processing of matrix multiplications and convolutions
- Significantly faster tensor operations

**Advantages:**
- 20-30x faster training
- Real-time visual feedback during training
- Efficient batch processing
- Lower power consumption than CUDA GPUs

**Requirements:**
- Apple Silicon Mac (M1, M2, M3, M4 series)
- PyTorch with MPS support (included in requirements.txt)

**Code Implementation:**
```python
# Backend automatically detects and uses MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Falls back to NVIDIA GPU
```

#### NVIDIA GPU Acceleration (CUDA)

**How it works:**
- Uses NVIDIA CUDA backend
- Supports GTX 10xx series and newer (RTX 20xx/30xx/40xx recommended)
- Typically 20-40x faster than CPU
- Often faster than Apple Silicon MPS

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA drivers installed
- PyTorch with CUDA support

**Setup:** See [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md) for detailed Windows/NVIDIA installation instructions

**Code Implementation:**
```python
# Backend automatically detects and uses CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
```

#### CPU Training

**How it works:**
- All computations run on CPU cores
- Sequential or limited parallel processing
- Standard PyTorch CPU backend

**Advantages:**
- Works on any machine
- No special hardware required
- Good for learning and understanding the algorithm

**Use cases:**
- Testing and development
- Machines without GPU
- Educational purposes where speed is not critical

### Algorithm-Level Optimizations

Both GPU and CPU benefit from:

1. **Batch Processing**: Process 64 images simultaneously
   - Reduces overhead per sample
   - Better utilizes parallel hardware

2. **Async Event Loop**: Training runs asynchronously
   - WebSocket connections stay alive
   - UI remains responsive
   - `await asyncio.sleep(0)` yields control after each batch

3. **Efficient Data Loading**:
   - Images normalized to [-1, 1]
   - Converted to RGB format
   - Cached in memory during epoch

4. **Optimized Network Architecture**:
   - Strided convolutions (no pooling)
   - Batch normalization for stable training
   - LeakyReLU for discriminator (prevents dead neurons)

### Performance Monitoring

The UI shows real-time performance metrics:
- **Batch updates**: Every 5 batches on GPU, 10 on CPU
- **Epoch completion time**: Displayed in logs
- **Device indicator**: Shows current compute device
- **Progress bar**: Visual training progress

## API Endpoints

### REST API

- `GET /` - API information
- `GET /status` - Get training status and current device
- `GET /metrics` - Get all training metrics
- `POST /start_training` - Start training with config (includes device selection)
- `POST /stop_training` - Stop ongoing training
- `POST /generate` - Generate synthetic images
- `POST /save_model` - Save current model checkpoint
- `POST /load_model` - Load saved model checkpoint

### WebSocket

- `WS /ws` - Real-time training updates with heartbeat mechanism

## Understanding the Metrics

### Loss Curves
- **Generator Loss (L_G)**: Should generally decrease but may fluctuate
- **Discriminator Loss (L_D)**: Should stabilize around 0.5-0.7
- If D loss → 0: Discriminator too strong (generator can't fool it)
- If G loss → 0 but D loss high: Mode collapse potential

### Discriminator Scores
- **D(real)**: Should stay close to 1.0 (correctly identifies real images)
- **D(fake)**: Should increase from ~0 toward 0.5 as G improves
- At equilibrium: Both converge to ~0.5 (generator creates perfect fakes)

## Training Tips

1. **Start with MNIST**: Easier to train, converges faster than Fashion-MNIST
2. **Use GPU (MPS) if available**: 20-30x faster than CPU
   - Apple Silicon Macs: Select "GPU (Apple Silicon MPS)"
   - Other systems: Select "CPU"
3. **Monitor D(real) and D(fake)**:
   - If D(real) < 0.8: Discriminator struggling
   - If D(fake) stays near 0: Generator not improving (check learning rate)
   - Both should converge toward 0.5 at equilibrium
4. **Typical training time**:
   - GPU (M4 Max): 10-30 minutes for 10 epochs
   - CPU: 3-8 hours for 10 epochs
   - 5-10 epochs needed for visible results
5. **Save your models**: Use "Save Model" after successful training
6. **Experiment with hyperparameters**:
   - Larger batch size (128, 256) = faster but more memory
   - Higher learning rate (0.0003-0.0005) = faster convergence but less stable
   - More epochs (20-50) = better quality images

## Project Structure

```
dcgan-demo/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── models.py         # Generator & Discriminator
│   ├── trainer.py        # Training logic
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   ├── App.css       # Styles
│   │   ├── main.jsx      # React entry point
│   │   └── index.css     # Global styles
│   ├── package.json      # Node dependencies
│   ├── vite.config.js    # Vite configuration
│   └── index.html        # HTML template
└── README.md
```

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Find and kill process using port 8000:
lsof -ti:8000 | xargs kill -9

# Or change port in main.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**GPU (MPS) not detected on Apple Silicon:**
```bash
# Check PyTorch MPS support:
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, update PyTorch:
pip install --upgrade torch torchvision
```

**MPS errors during training:**
```bash
# Some operations may not be fully supported on MPS
# Switch to CPU via UI or edit main.py:
device = torch.device("cpu")
```

**Dataset download fails:**
```bash
# The trainer will auto-download MNIST to ./data/ directory
# If download fails, check internet connection
# Or manually download from: http://yann.lecun.com/exdb/mnist/
```

**Training stuck at epoch 1:**
- Check backend terminal for errors
- Try restarting backend
- Switch to CPU if MPS is causing issues

### Frontend Issues

**WebSocket disconnections:**
- Backend may be blocked on slow CPU training
- Switch to GPU (MPS) for faster iterations
- Backend logs show connection status

**WebSocket connection failed:**
- Ensure backend is running on port 8000
- Check CORS settings in main.py
- Verify `WS_URL` in App.jsx matches backend

**Charts not displaying:**
```bash
# Reinstall dependencies:
npm install recharts
```

**Port 5173 in use:**
```bash
# Edit vite.config.js and change port:
server: { port: 3000 }
```

### Performance Issues

**Training very slow:**
- **On Apple Silicon**: Ensure MPS (GPU) is selected, not CPU
- **On other systems**: CPU training is slow; consider using Google Colab with GPU
- Reduce batch size or epochs for faster testing

**UI freezing:**
- Backend may be blocking the event loop
- Check that `await asyncio.sleep(0)` is present after each batch
- Restart backend if issue persists

## Mathematical Background

### Minimax Objective
```
min_G max_D V(D,G) = 𝔼_x~p_data[log D(x)] + 𝔼_z~p_z[log(1 - D(G(z)))]
```

### Generator Loss (Non-Saturating)
```
L_G = -𝔼_z~p_z[log D(G(z))]
```

### Discriminator Loss
```
L_D = -𝔼_x~p_data[log D(x)] - 𝔼_z~p_z[log(1 - D(G(z)))]
```

## References

- **Original GAN Paper**: Goodfellow et al., "Generative Adversarial Networks" (2014)
- **DCGAN Paper**: Radford et al., "Unsupervised Representation Learning with DCGANs" (2015)
- **PyTorch Tutorial**: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## License

Educational purposes - GCU AIT-204 Course

## Author

Created for AIT-204 GAN Lecture and Demonstration
