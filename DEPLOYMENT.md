# Cloud Deployment Guide

This guide explains how to deploy the DCGAN Demo application to the cloud.

## Recommended: Deploy Everything on Render (Easiest)

Render can host both frontend and backend together with one click using the Blueprint file.

### One-Click Deploy to Render

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **"New +"** → **"Blueprint"**
4. Connect your GitHub repository
5. Render will detect the `render.yaml` file and create both services
6. Click **"Apply"** to deploy

That's it! Render will:
- Build and deploy the Python backend (FastAPI + PyTorch)
- Build and deploy the React frontend (Vite static site)
- Automatically link them together

Your app will be available at:
- Frontend: `https://dcgan-frontend.onrender.com`
- Backend API: `https://dcgan-api.onrender.com`

---

## Alternative: Separate Deployment

If you prefer more control, you can deploy frontend and backend separately.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Vercel        │  HTTP   │   Render        │
│   (Frontend)    │ ──────► │   (Backend)     │
│   React + Vite  │         │   FastAPI       │
└─────────────────┘         └─────────────────┘
```

## Prerequisites

1. GitHub account with your code pushed to a repository
2. Render account (https://render.com)
3. Vercel account (https://vercel.com) - optional

---

## Option A: Deploy Backend to Render

### Option A: Using Render Dashboard

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `dcgan-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   - `ALLOWED_ORIGINS`: `*` (or your Vercel frontend URL)
6. Click **"Create Web Service"**

### Option B: Using render.yaml (Blueprint)

1. Push the `backend/render.yaml` file to your repo
2. Go to Render Dashboard → **"Blueprints"**
3. Connect your repo and deploy

### Important Notes for Render

- **Free tier** will spin down after 15 minutes of inactivity
- First request after spin-down takes ~30-60 seconds
- PyTorch is large (~2GB), so builds may take 5-10 minutes
- Free tier uses CPU only (no GPU)

---

## Step 2: Deploy Frontend to Vercel

### Option A: Using Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to frontend directory
cd frontend

# Deploy
vercel

# Follow the prompts:
# - Link to existing project? No
# - Project name: dcgan-frontend
# - Directory: ./
# - Override settings? No
```

### Option B: Using Vercel Dashboard

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Add Environment Variables:
   - `VITE_API_URL`: `https://your-backend-name.onrender.com`
   - `VITE_WS_URL`: `wss://your-backend-name.onrender.com/ws`
6. Click **"Deploy"**

---

## Step 3: Configure Environment Variables

### Backend (Render)

| Variable | Value | Description |
|----------|-------|-------------|
| `ALLOWED_ORIGINS` | `https://your-frontend.vercel.app` | CORS allowed origins |

### Frontend (Vercel)

| Variable | Value | Description |
|----------|-------|-------------|
| `VITE_API_URL` | `https://dcgan-backend.onrender.com` | Backend API URL |
| `VITE_WS_URL` | `wss://dcgan-backend.onrender.com/ws` | WebSocket URL |

---

## Step 4: Update CORS After Frontend Deploy

Once you have your Vercel URL (e.g., `https://dcgan-demo.vercel.app`):

1. Go to Render Dashboard → Your Service → Environment
2. Update `ALLOWED_ORIGINS` to your Vercel URL
3. Redeploy the backend

---

## Troubleshooting

### Backend Issues

**Build fails with "No space left on device"**
- PyTorch is large. Consider using a paid tier or using `torch-cpu` version

**WebSocket disconnects frequently**
- Free tier may timeout. Consider upgrading for persistent connections

**Training is slow**
- Free tier is CPU-only. For GPU, use Render's paid GPU instances or other providers

### Frontend Issues

**API calls fail with CORS error**
- Verify `ALLOWED_ORIGINS` includes your Vercel URL
- Make sure you're using `https://` not `http://`

**WebSocket won't connect**
- Use `wss://` (secure) instead of `ws://` for production
- Check browser console for specific errors

---

## Alternative: Deploy Both on Render

You can also deploy the frontend as a static site on Render:

1. Create a new **Static Site** on Render
2. Root Directory: `frontend`
3. Build Command: `npm install && npm run build`
4. Publish Directory: `dist`
5. Add the same environment variables

---

## Cost Considerations

### Free Tier Limitations

| Service | Render Free | Vercel Free |
|---------|-------------|-------------|
| Compute | 750 hrs/month | Unlimited |
| Memory | 512 MB | N/A |
| Sleep | After 15 min idle | Never |
| GPU | Not available | N/A |
| Bandwidth | 100 GB/month | 100 GB/month |

### For Better Performance

- **Render Starter** ($7/month): No sleep, more memory
- **Render GPU** ($XX/hour): For actual GPU training
- Consider **Railway**, **Fly.io**, or **AWS** for GPU workloads

---

## Local Development After Cloud Setup

When developing locally, the app will automatically use `localhost`:

```bash
# Frontend uses VITE_API_URL env var, defaults to localhost
cd frontend
npm run dev

# Backend runs locally
cd backend
./venv/bin/python main.py
```

No changes needed - the environment variables only apply in production.
