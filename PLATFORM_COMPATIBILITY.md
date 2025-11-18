# Platform Compatibility Guide

Quick reference for running DCGAN demo on different platforms.

## ✅ Supported Platforms

| Platform | GPU Support | Status | Setup Guide |
|----------|------------|--------|-------------|
| **macOS (Apple Silicon)** | MPS (Metal) | ✅ Works out of box | Included in main setup |
| **Windows (NVIDIA GPU)** | CUDA | ✅ Fully supported | [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md) |
| **Linux (NVIDIA GPU)** | CUDA | ✅ Fully supported | [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md) |
| **Windows (Intel/AMD CPU)** | CPU only | ✅ Works (slow) | Standard setup |
| **macOS (Intel)** | CPU only | ✅ Works (slow) | Standard setup |
| **Linux (CPU only)** | CPU only | ✅ Works (slow) | Standard setup |

---

## 🚀 Performance Comparison

### Training 10 Epochs of MNIST (Batch Size 64)

| Setup | Device | Time | Speedup | Best For |
|-------|--------|------|---------|----------|
| **Windows + RTX 3060** | CUDA | 5-10 min | **40x** | ⭐ Best performance |
| **Mac M4 Max** | MPS | 10-20 min | **30x** | ⭐ Mac users |
| **Windows + GTX 1660** | CUDA | 10-20 min | **20x** | Good performance |
| **Linux + RTX 2060** | CUDA | 8-15 min | **30x** | Good performance |
| **Any CPU (i7/i9)** | CPU | 3-5 hours | 1x | Testing only |

---

## 🔧 Setup Summary

### Mac (Apple Silicon)

**Requirements:**
- M1, M2, M3, or M4 chip
- macOS 12.3 or later

**Setup:**
```bash
./start_backend.sh  # Automatically detects MPS
```

**Device selection in UI:**
- Choose: "GPU (Metal/MPS - Mac)"

---

### Windows (NVIDIA GPU)

**Requirements:**
- NVIDIA GPU (GTX 10xx or newer)
- CUDA drivers installed

**Setup:**
```cmd
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then run normally
start_backend.bat
```

**Device selection in UI:**
- Choose: "GPU (CUDA - NVIDIA)"

**Full guide:** [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md)

---

### Linux (NVIDIA GPU)

**Requirements:**
- NVIDIA GPU (GTX 10xx or newer)
- CUDA drivers installed

**Setup:**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then run normally
./start_backend.sh
```

**Device selection in UI:**
- Choose: "GPU (CUDA - NVIDIA)"

**Full guide:** [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md)

---

### CPU Only (Any Platform)

**Works on:**
- Any computer without GPU
- Intel Mac
- Windows/Linux without NVIDIA GPU

**Setup:**
```bash
# Standard installation
pip install -r requirements.txt
```

**Device selection in UI:**
- Choose: "CPU"

**Performance tips:**
- Reduce epochs to 5 for testing
- Use smaller batch size (32)
- Be patient - each epoch takes 20-50 minutes

---

## 🎯 For Students

### Before Class Checklist

**1. Check your hardware:**
```bash
# Mac users:
system_profiler SPHardwareDataType | grep "Chip"

# Windows/Linux users:
nvidia-smi  # Should show GPU if you have NVIDIA
```

**2. Verify PyTorch GPU support:**
```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
- **Mac M1/M2/M3/M4:** `MPS: True, CUDA: False`
- **Windows/Linux NVIDIA:** `MPS: False, CUDA: True`
- **CPU only:** `MPS: False, CUDA: False`

**3. Choose your setup:**
- ✅ MPS available → Use standard setup
- ✅ CUDA available → Follow [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md)
- ⚠️ Neither available → Use CPU (slow but works)

---

## 🐛 Quick Troubleshooting

### "Using CPU" but I have a GPU

**Mac:**
```bash
# Check MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
# If False, update PyTorch:
pip install --upgrade torch torchvision
```

**Windows/Linux:**
```bash
# Check CUDA support
python -c "import torch; print(torch.cuda.is_available())"
# If False, you need CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training is slow even with GPU selected

**Check actual device in use:**
- Look at backend logs when starting training
- Should say "Using NVIDIA GPU (CUDA)" or "Using Apple Silicon GPU (MPS)"
- If says "Using CPU", see above

**Monitor GPU usage:**
```bash
# Windows/Linux:
nvidia-smi -l 1  # Updates every second, should show 70-100% GPU usage

# Mac:
sudo powermetrics --samplers gpu_power -i 1000  # Should show GPU active
```

### Out of memory errors

**Reduce batch size in UI:**
- 4GB VRAM: batch_size = 16
- 6GB VRAM: batch_size = 32
- 8GB+ VRAM: batch_size = 64

---

## 📊 Recommended Settings by Platform

### NVIDIA RTX 30xx/40xx (High-end)
```
Device: GPU (CUDA - NVIDIA)
Batch Size: 128
Epochs: 10
Learning Rate: 0.0002
Expected Time: ~3-5 minutes
```

### NVIDIA GTX 16xx/20xx (Mid-range)
```
Device: GPU (CUDA - NVIDIA)
Batch Size: 64
Epochs: 10
Learning Rate: 0.0002
Expected Time: ~10-15 minutes
```

### Apple M3/M4 (Latest Mac)
```
Device: GPU (Metal/MPS - Mac)
Batch Size: 64
Epochs: 10
Learning Rate: 0.0002
Expected Time: ~10-20 minutes
```

### Apple M1/M2 (Older Mac)
```
Device: GPU (Metal/MPS - Mac)
Batch Size: 64
Epochs: 10
Learning Rate: 0.0002
Expected Time: ~15-30 minutes
```

### CPU Only
```
Device: CPU
Batch Size: 32
Epochs: 5 (for testing)
Learning Rate: 0.0002
Expected Time: ~1-2 hours
```

---

## 🎓 For Instructors

### Class Preparation

**Survey students:**
1. Count students with NVIDIA GPUs (fastest)
2. Count students with Apple Silicon Macs (fast)
3. Count students with CPU only (slow)

**Pre-class setup:**
- Students with NVIDIA: Share NVIDIA_GPU_SETUP.md early
- Students with CPU only: Set expectations (slow but works)
- All students: Run `python diagnose.py` before class

**In-class strategy:**
- GPU students: Full 10 epoch training
- CPU students: 5 epochs for demo, use pre-trained models for experiments

**Provide pre-trained models:**
- Save models trained on GPU
- CPU students can load and experiment without long waits

---

## 📚 Additional Resources

- **Main Documentation:** [README.md](README.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **NVIDIA Setup:** [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md)
- **Student Project:** [STUDENT_PROJECT.html](STUDENT_PROJECT.html)

---

## ✨ Summary

| If you have... | Use this device | Expected speed | Setup needed |
|---------------|-----------------|----------------|--------------|
| Mac M1/M2/M3/M4 | GPU (MPS) | ⚡⚡ Very Fast | None - works out of box |
| NVIDIA RTX GPU | GPU (CUDA) | ⚡⚡⚡ Fastest | [Follow guide](NVIDIA_GPU_SETUP.md) |
| NVIDIA GTX GPU | GPU (CUDA) | ⚡⚡ Fast | [Follow guide](NVIDIA_GPU_SETUP.md) |
| Only CPU | CPU | 🐌 Slow | None - reduce epochs |

**Bottom line:** The app works on all platforms. GPU users get 20-40x speedup!
