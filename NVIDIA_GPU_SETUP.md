# Running DCGAN with NVIDIA GPU on Windows

This guide explains how to set up and run the DCGAN demo on Windows machines with NVIDIA GPUs using CUDA.

## TL;DR - Does it Work?

**YES!** The code already supports NVIDIA GPUs. The backend automatically detects and uses CUDA if available:

```python
# From main.py - already implemented!
if torch.backends.mps.is_available():
    device = torch.device("mps")      # Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")     # NVIDIA GPU
else:
    device = torch.device("cpu")      # CPU fallback
```

However, you need to install CUDA-enabled PyTorch and make minor UI adjustments.

---

## Prerequisites

### 1. Check Your GPU

First, verify you have an NVIDIA GPU:

**Windows:**
```cmd
# Open Command Prompt and run:
nvidia-smi
```

You should see output showing:
- GPU model (e.g., RTX 3060, GTX 1660)
- CUDA Version
- GPU memory

**Example output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
```

### 2. Install CUDA Toolkit (if needed)

Most modern NVIDIA drivers include CUDA support. If `nvidia-smi` shows a CUDA version, you're good!

If not, download from: https://developer.nvidia.com/cuda-downloads

**Recommended versions:**
- CUDA 11.8 or 12.x
- Matching cuDNN (usually included with PyTorch)

---

## Installation Steps

### Option 1: Quick Setup (Recommended)

1. **Modify `backend/requirements.txt`** to use CUDA-enabled PyTorch:

```txt
# Replace the torch and torchvision lines with:
--index-url https://download.pytorch.org/whl/cu118
torch>=2.0.0
torchvision>=0.15.0

# Or for CUDA 12.1:
--index-url https://download.pytorch.org/whl/cu121
torch>=2.0.0
torchvision>=0.15.0

# Keep other dependencies:
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
websockets>=12.0
Pillow>=10.1.0
numpy>=1.24.0
pydantic>=2.0.0
```

2. **Create virtual environment and install:**

```cmd
# Navigate to backend directory
cd dcgan-demo\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Verify CUDA installation:**

```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3060
```

### Option 2: Manual PyTorch Installation

If requirements.txt doesn't work, install PyTorch manually:

**Visit:** https://pytorch.org/get-started/locally/

**Select:**
- PyTorch Build: Stable
- Your OS: Windows
- Package: Pip
- Language: Python
- Compute Platform: CUDA 11.8 (or your version)

**Example command:**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install other dependencies:
```cmd
pip install fastapi uvicorn[standard] websockets Pillow numpy pydantic python-multipart
```

---

## Code Modifications

### 1. Update Frontend Device Selector (Optional)

Make the UI more generic by updating `frontend/src/App.jsx`:

**Current code (line ~286):**
```jsx
<option value="mps">GPU (Apple Silicon MPS)</option>
<option value="cpu">CPU</option>
```

**Updated code:**
```jsx
<option value="mps">GPU (Metal/MPS)</option>
<option value="cuda">GPU (NVIDIA CUDA)</option>
<option value="cpu">CPU</option>
```

**Also update the hint text (line ~289):**
```jsx
<small style={{display: 'block', marginTop: '4px', color: '#666'}}>
  {config.device === 'mps' ? '⚡ Fast training (~1-3 min/epoch)' :
   config.device === 'cuda' ? '⚡ Fast training (~1-3 min/epoch)' :
   '🐌 Slow training (~20-50 min/epoch)'}
</small>
```

**And update the device indicator (line ~414):**
```jsx
<p>
  <strong>Device:</strong>{' '}
  {status.device === 'mps' ? '⚡ GPU (Metal)' :
   status.device === 'cuda' ? '⚡ GPU (CUDA)' :
   '🖥️ CPU'}
</p>
```

### 2. Verify Backend Compatibility

The backend code already supports CUDA! Verify `backend/main.py` has:

```python
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
```

✅ This is already correct - no changes needed!

### 3. Update Diagnostics (Optional)

Update `backend/diagnose.py` to be more informative about CUDA:

Find the `check_pytorch()` function and ensure it includes:

```python
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
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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
```

✅ This is already implemented correctly!

---

## Running the Application

### Start Backend (Windows)

```cmd
cd dcgan-demo\backend
venv\Scripts\activate
python main.py
```

**Look for this in the output:**
```
INFO:     Started server process
INFO:     Using NVIDIA GPU (CUDA)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Frontend (Windows)

```cmd
cd dcgan-demo\frontend
npm install
npm run dev
```

**Open browser:** http://localhost:5173

---

## Performance Comparison

### Training Speed (10 epochs, MNIST, batch size 64)

| Device | Time per Epoch | Total Time | Speedup |
|--------|---------------|------------|---------|
| **NVIDIA RTX 3060** | 30-60 seconds | 5-10 minutes | **30-40x** |
| **NVIDIA GTX 1660** | 1-2 minutes | 10-20 minutes | **15-20x** |
| **Apple M4 Max (MPS)** | 1-3 minutes | 10-30 minutes | **20-30x** |
| **CPU (Intel i7)** | 20-50 minutes | 200-500 minutes | 1x (baseline) |

**NVIDIA GPUs are typically FASTER than Apple Silicon MPS!**

---

## Troubleshooting

### "CUDA out of memory"

**Solution:** Reduce batch size in the UI:
```
Change: Batch Size = 64
To:     Batch Size = 32 or 16
```

NVIDIA GPUs have limited VRAM. If you have:
- **4GB VRAM:** Use batch_size = 16-32
- **6GB VRAM:** Use batch_size = 32-64
- **8GB+ VRAM:** Use batch_size = 64-128

### "CUDA not available" but nvidia-smi works

**Cause:** Wrong PyTorch version (CPU-only)

**Solution:** Reinstall PyTorch with CUDA:
```cmd
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "RuntimeError: Couldn't load custom C++ ops"

**Cause:** cuDNN not properly installed

**Solution:** Update PyTorch:
```cmd
pip install --upgrade torch torchvision
```

### Training slower than expected on GPU

**Check:**
1. Verify GPU is being used:
   ```python
   python -c "import torch; x = torch.randn(1000, 1000).cuda(); print('GPU working!')"
   ```

2. Monitor GPU usage during training:
   ```cmd
   # In another terminal:
   nvidia-smi -l 1  # Update every 1 second
   ```

   You should see:
   - GPU utilization: 70-100%
   - Memory usage: Several GB

3. Check device in UI - should show "⚡ GPU (CUDA)" not "🖥️ CPU"

### Windows Firewall Blocks Server

**Symptom:** Frontend can't connect to backend

**Solution:**
```cmd
# Allow Python through firewall:
netsh advfirewall firewall add rule name="Python" dir=in action=allow program="C:\path\to\python.exe"
```

Or temporarily disable firewall for testing.

---

## Common Issues & Solutions

### Issue: PyTorch installs but CUDA still unavailable

**Diagnosis:**
```cmd
python -c "import torch; print(torch.version.cuda)"
```

If output is `None`, you have CPU-only PyTorch.

**Fix:** Use the index URL when installing:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Multiple CUDA versions on system

**Diagnosis:**
```cmd
nvidia-smi  # Shows driver CUDA version
python -c "import torch; print(torch.version.cuda)"  # Shows PyTorch CUDA version
```

**Solution:** These can differ slightly. PyTorch's CUDA version should be ≤ driver version.

**Example:**
- Driver CUDA: 12.2 ✓
- PyTorch CUDA: 11.8 ✓ (this is fine!)

### Issue: Training starts on CPU instead of GPU

**Check backend logs:**
```
INFO:     Using CPU  ← Wrong!
```

Should be:
```
INFO:     Using NVIDIA GPU (CUDA)  ← Correct!
```

**Fix:** Run diagnostics:
```cmd
cd backend
python diagnose.py
```

---

## Performance Optimization Tips

### 1. Increase Batch Size
If you have enough VRAM (8GB+), use larger batches:
```
Batch size: 128 or 256
```
Result: Faster training, better GPU utilization

### 2. Mixed Precision Training (Advanced)

For even faster training on modern GPUs (RTX 20xx/30xx/40xx):

**Modify `trainer.py`** to use automatic mixed precision:

```python
# At the top of trainer.py
from torch.cuda.amp import autocast, GradScaler

# In __init__ method
self.scaler = GradScaler() if device.type == 'cuda' else None

# In train_step method
def train_step(self, real_images):
    batch_size = real_images.size(0)
    real_label = 1.0
    fake_label = 0.0

    # Update Discriminator with mixed precision
    self.netD.zero_grad()

    if self.scaler:  # Use AMP on CUDA
        with autocast():
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
            output = self.netD(real_images).view(-1)
            errD_real = self.criterion(output, label)

        self.scaler.scale(errD_real).backward()
        # ... rest of training
        self.scaler.step(self.optimizerD)
        self.scaler.update()
    else:  # Standard training on CPU/MPS
        # ... existing code
```

**Speedup:** 1.5-2x faster on Tensor Core GPUs

### 3. Pin Memory for Faster Data Loading

Already implemented in `trainer.py:117`:
```python
pin_memory=True if self.device.type == 'cuda' else False
```

✓ No changes needed!

---

## Startup Script for Windows

Create `start_backend.bat` in `dcgan-demo/` directory:

```batch
@echo off
echo Starting DCGAN Backend Server...
cd backend

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Running diagnostics...
python diagnose.py

if errorlevel 1 (
    echo.
    echo Diagnostics failed! Please fix the errors above.
    echo If you want to try anyway, run: python main.py
    pause
    exit /b 1
)

echo.
echo Diagnostics passed!
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop
echo.
python main.py
```

**Usage:**
```cmd
# Double-click start_backend.bat or run:
start_backend.bat
```

---

## Quick Reference

### Check GPU Status
```cmd
# See GPU info
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Monitor GPU during training
nvidia-smi -l 1
```

### Reinstall PyTorch with CUDA
```cmd
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Test CUDA Performance
```cmd
python -c "import torch; x = torch.randn(10000, 10000).cuda(); y = x @ x; print(f'GPU test passed! Device: {y.device}')"
```

---

## Summary

✅ **Good News:** The code already works with NVIDIA GPUs!

✅ **What's Needed:**
1. Install CUDA-enabled PyTorch (`--index-url` flag)
2. (Optional) Update frontend to show "CUDA" instead of "MPS"
3. (Optional) Adjust batch size for your GPU memory

✅ **Performance:** NVIDIA GPUs are typically 20-40x faster than CPU

✅ **Compatibility:** Works on any NVIDIA GPU with CUDA support (GTX 10xx series or newer)

---

## For Instructors

When teaching this to students on Windows:

1. **Before class:** Have students run diagnostics
   ```cmd
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Provide pre-configured** `requirements.txt` with CUDA PyTorch

3. **Expect variations:** Different GPUs = different speeds
   - RTX 40xx: Very fast (< 5 min for 10 epochs)
   - RTX 30xx/20xx: Fast (5-10 min)
   - GTX 16xx: Moderate (10-20 min)
   - GTX 10xx: Slower (15-30 min)

4. **Have CPU fallback ready** for students without GPUs

5. **Memory issues:** Keep batch_size=32 as default for students

---

## Additional Resources

- **PyTorch CUDA Guide:** https://pytorch.org/get-started/locally/
- **NVIDIA CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **GPU Compatibility:** https://developer.nvidia.com/cuda-gpus
- **PyTorch Forums:** https://discuss.pytorch.org/

---

**Questions?** Check the main README.md or consult PyTorch documentation.

**Happy Training! 🚀**
