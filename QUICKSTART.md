# DCGAN Demo - Quick Start Guide

Get the DCGAN demo running in 5 minutes!

## Option 1: Using Shell Scripts (Mac/Linux)

### Terminal 1 - Start Backend
```bash
./start_backend.sh
```

### Terminal 2 - Start Frontend
```bash
./start_frontend.sh
```

Then open your browser to `http://localhost:5173`

---

## Option 2: Manual Setup

### Step 1: Backend Setup

```bash
# Navigate to backend
cd dcgan-demo/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Frontend Setup

**In a new terminal:**

```bash
# Navigate to frontend
cd dcgan-demo/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

**Expected output:**
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 3: Open the App

Open your browser to: `http://localhost:5173`

---

## Using the Demo

### 1. Start Training

1. Select dataset (MNIST or Fashion-MNIST)
2. **Choose device** (GPU or CPU) - GPU is 20-30x faster!
3. Set epochs (try 10 for quick demo)
4. Click **"Start Training"**
5. Watch the magic happen:
   - Loss curves update in real-time
   - Sample images generated each epoch
   - Logs show progress
   - Device indicator shows GPU/CPU usage

### 2. Save Your Model

After training completes:
1. Click **"Save Model"** to save the trained weights
2. Model saved as `dcgan_checkpoint.pth` in backend directory
3. Can load later to continue or generate more images

### 3. Generate Images

1. Click **"Generate 16 Images"** or **"Generate 64 Images"**
2. See synthetic images created by the trained generator
3. Can generate unlimited images from saved model

### 4. Load Saved Model

1. Click **"Load Model"** to restore previously saved checkpoint
2. Metrics and epoch count will be restored
3. Can continue training or just generate images

### 3. Interpret Results

**Good Training Signs:**
- Generator loss decreases over time
- Discriminator loss stabilizes around 0.5-0.7
- D(real) stays near 1.0
- D(fake) increases toward 0.5
- Generated images become clearer each epoch

**Problems:**
- D loss → 0: Discriminator too strong (try lower learning rate for D)
- G loss stays high: Generator can't fool D (try more epochs)
- Both losses oscillate wildly: Training instability (lower learning rate)

---

## Training Time Estimates

### NVIDIA GPU (CUDA - Windows/Linux)
- **MNIST, 10 epochs**: ~5-10 minutes ⚡⚡ (FASTEST!)
- **Fashion-MNIST, 10 epochs**: ~10-20 minutes ⚡⚡

### Apple Silicon (MPS - Mac)
- **MNIST, 10 epochs**: ~10-20 minutes ⚡
- **Fashion-MNIST, 10 epochs**: ~15-30 minutes ⚡

### CPU (Any System)
- **MNIST, 10 epochs**: ~3-5 hours 🐌
- **Fashion-MNIST, 10 epochs**: ~4-8 hours 🐌

**Recommendations:**
- **Windows/Linux with NVIDIA GPU:** See [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md) for setup
- **Mac with Apple Silicon:** Use GPU (MPS) option
- **Any system without GPU:** Reduce epochs to 5 for faster testing

---

## Troubleshooting

### Backend won't start

**Check Python version:**
```bash
python3 --version  # Should be 3.8+
```

**Install PyTorch manually if needed:**
```bash
pip install torch torchvision
```

### Frontend won't start

**Check Node version:**
```bash
node --version  # Should be 16+
```

**Clear npm cache:**
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### WebSocket not connecting

1. Ensure backend is running on port 8000
2. Check browser console for errors
3. Try refreshing the page

### Training is too slow

1. **Switch to GPU**: Select "GPU (Apple Silicon MPS)" in device dropdown
2. **Reduce batch size**: Try 32 instead of 64
3. **Fewer epochs**: Start with 5 epochs for testing
4. **Check device**: Verify GPU indicator shows "⚡ GPU (MPS)" not "🖥️ CPU"

---

## Example Training Session

```
1. Start backend and frontend
2. Select MNIST dataset
3. Choose GPU (Apple Silicon MPS)
4. Set 10 epochs, batch size 64
5. Click "Start Training"
6. After ~15 minutes (GPU):
   - Generator creates digit-like shapes
   - Losses stabilize
   - D(fake) increases from 0.1 to 0.4+
   - Sample images improve each epoch
7. Click "Save Model"
8. Click "Generate 64 Images"
9. See 64 synthetic handwritten digits!
```

---

## Next Steps

- **Experiment with hyperparameters**: Try different learning rates, batch sizes
- **Compare devices**: Train same model on GPU vs CPU and compare results
- **Compare datasets**: Train on MNIST vs Fashion-MNIST
- **Save and share**: Save your best models and share generated images
- **Advanced training**: Try 50+ epochs for higher quality results

---

## Need Help?

Check the main `README.md` for:
- Detailed architecture explanation
- Mathematical formulations
- API documentation
- Advanced configuration options
