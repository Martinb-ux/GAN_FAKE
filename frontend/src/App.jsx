import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

// Use environment variables for production, fallback to localhost for development
const getApiUrl = () => {
  const envUrl = import.meta.env.VITE_API_URL;
  if (!envUrl) return 'http://localhost:8000';
  // Handle Render's host format (without protocol)
  if (envUrl.startsWith('http')) return envUrl;
  return `https://${envUrl}`;
};

const getWsUrl = () => {
  const envUrl = import.meta.env.VITE_WS_URL;
  if (!envUrl) return 'ws://localhost:8000/ws';
  // Handle Render's host format (without protocol)
  if (envUrl.startsWith('ws')) return envUrl;
  return `wss://${envUrl}/ws`;
};

const API_BASE = getApiUrl();
const WS_URL = getWsUrl();

function App() {
  const [status, setStatus] = useState({ is_training: false, current_epoch: 0, device: 'mps' });
  const [config, setConfig] = useState({
    dataset: 'mnist',
    epochs: 10,
    batch_size: 64,
    learning_rate: 0.0002,
    device: 'mps'
  });
  const [metrics, setMetrics] = useState({
    g_losses: [],
    d_losses: [],
    real_scores: [],
    fake_scores: []
  });
  const [currentImage, setCurrentImage] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [logs, setLogs] = useState([]);
  const [connected, setConnected] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Comparison dashboard state
  const [availableModels, setAvailableModels] = useState([]);
  const [trainingSessions, setTrainingSessions] = useState({});
  const [mnistImage, setMnistImage] = useState(null);
  const [fashionImage, setFashionImage] = useState(null);
  const [isLoadingComparison, setIsLoadingComparison] = useState(false);
  const [showComparison, setShowComparison] = useState(false);

  const wsRef = useRef(null);
  const logsEndRef = useRef(null);

  // WebSocket connection
  useEffect(() => {
    connectWebSocket();
    // Fetch available models on mount
    fetchAvailableModels();
    fetchTrainingSessions();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      addLog('Connected to training server');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
      addLog('WebSocket error - reconnecting...', 'error');
      setTimeout(connectWebSocket, 3000);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnected(false);
      addLog('Disconnected from server', 'warning');
      setTimeout(connectWebSocket, 3000);
    };

    wsRef.current = ws;
  };

  const handleWebSocketMessage = (data) => {
    console.log('WebSocket message:', data); // Debug logging

    switch (data.type) {
      case 'connected':
        addLog(data.message, 'success');
        if (data.status) {
          setStatus(data.status);
        }
        break;

      case 'batch_update':
        console.log('📊 Batch update - epoch:', data.epoch, 'batch:', data.batch);
        addLog(
          `Epoch ${data.epoch + 1}, Batch ${data.batch}/${data.total_batches} - ` +
          `D Loss: ${data.metrics.loss_d.toFixed(4)}, G Loss: ${data.metrics.loss_g.toFixed(4)}`
        );
        setStatus(prev => {
          const newStatus = { ...prev, is_training: true, current_epoch: data.epoch + 1 };
          console.log('Status updated (batch):', newStatus);
          return newStatus;
        });
        break;

      case 'epoch_complete':
        console.log('✅ Epoch complete - epoch:', data.epoch, 'metrics:', data.metrics);
        addLog(
          `✓ Epoch ${data.epoch + 1} completed - ` +
          `D Loss: ${data.metrics.d_loss.toFixed(4)}, G Loss: ${data.metrics.g_loss.toFixed(4)}`,
          'success'
        );
        if (data.sample_image) {
          console.log('🖼️ Received sample image, length:', data.sample_image.length);
          setCurrentImage(data.sample_image);
        } else {
          console.warn('⚠️ No sample image in epoch_complete message');
        }
        if (data.all_metrics) {
          console.log('📈 Updating metrics, g_losses length:', data.all_metrics.g_losses?.length);
          setMetrics(data.all_metrics);
        }
        setStatus(prev => {
          const newStatus = { is_training: true, current_epoch: data.epoch + 1 };
          console.log('Status updated (epoch):', newStatus);
          return newStatus;
        });
        setIsStarting(false);
        break;

      case 'training_complete':
        addLog(data.message, 'success');
        setStatus(prev => ({ is_training: false, current_epoch: prev.current_epoch }));
        setIsStarting(false);
        break;

      case 'error':
        addLog(`Error: ${data.message}`, 'error');
        setStatus(prev => ({ ...prev, is_training: false }));
        setIsStarting(false);
        setIsGenerating(false);
        break;
    }
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, type }].slice(-50)); // Keep last 50 logs
  };

  // Auto-scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Poll status as backup to WebSocket - but don't override WebSocket updates!
  useEffect(() => {
    const pollStatus = async () => {
      if (!connected) return; // Only poll if connected

      try {
        const response = await axios.get(`${API_BASE}/status`);
        setStatus(prev => {
          // Only update if backend shows different training state
          // Don't update epoch from polling - WebSocket is more accurate
          if (prev.is_training !== response.data.is_training) {
            console.log('Training status changed via poll:', response.data.is_training);
            return { ...prev, is_training: response.data.is_training };
          }
          return prev;
        });
      } catch (error) {
        console.error('Status poll error:', error);
      }
    };

    // Poll every 5 seconds (less frequent to not interfere with WebSocket)
    const interval = setInterval(pollStatus, 5000);
    return () => clearInterval(interval);
  }, [connected]);

  const startTraining = async () => {
    console.log('Start training clicked', config); // Debug
    setIsStarting(true);
    addLog('Starting training...', 'info');

    try {
      const response = await axios.post(`${API_BASE}/start_training`, config);
      console.log('Start training response:', response.data); // Debug

      if (response.data.success) {
        addLog('✓ Training started successfully - waiting for first batch...', 'success');
        setStatus(prev => ({ ...prev, is_training: true }));
      } else {
        addLog(response.data.message, 'warning');
        setIsStarting(false);
      }
    } catch (error) {
      console.error('Start training error:', error); // Debug
      addLog(`✗ Error starting training: ${error.message}`, 'error');
      setIsStarting(false);
    }
  };

  const stopTraining = async () => {
    console.log('Stop training clicked'); // Debug
    addLog('Stopping training...', 'warning');

    try {
      const response = await axios.post(`${API_BASE}/stop_training`);
      console.log('Stop training response:', response.data); // Debug

      if (response.data.success) {
        addLog('✓ Training stopped', 'warning');
        setStatus(prev => ({ ...prev, is_training: false }));
      }
    } catch (error) {
      console.error('Stop training error:', error); // Debug
      addLog(`✗ Error stopping training: ${error.message}`, 'error');
    }
  };

  const generateImages = async (numImages = 16) => {
    console.log('Generate images clicked:', numImages); // Debug
    setIsGenerating(true);
    addLog(`Generating ${numImages} images...`, 'info');

    try {
      const response = await axios.post(`${API_BASE}/generate`, {
        num_images: numImages
      });
      console.log('Generate response received'); // Debug

      if (response.data.success) {
        setGeneratedImage(response.data.image);
        addLog(`✓ Generated ${response.data.num_images} images`, 'success');
      }
    } catch (error) {
      console.error('Generate error:', error); // Debug
      addLog(`✗ Error generating images: ${error.message}`, 'error');
    } finally {
      setIsGenerating(false);
    }
  };

  const saveModel = async () => {
    setIsSaving(true);
    // Save with current dataset name
    const modelName = config.dataset;
    addLog(`Saving model as ${modelName}...`, 'info');

    try {
      const response = await axios.post(`${API_BASE}/save_model`, {
        model_name: modelName
      });

      if (response.data.success) {
        addLog(`✓ Model saved: ${response.data.path}`, 'success');
        // Refresh available models list
        fetchAvailableModels();
      }
    } catch (error) {
      addLog(`✗ Error saving model: ${error.message}`, 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const loadModel = async (modelName = null) => {
    setIsLoading(true);
    addLog(`Loading model${modelName ? ` (${modelName})` : ''}...`, 'info');

    try {
      const response = await axios.post(`${API_BASE}/load_model`, {
        model_name: modelName || 'default'
      });

      if (response.data.success) {
        addLog(`✓ Model loaded: ${response.data.path}`, 'success');
        // Refresh metrics and status
        if (response.data.metrics) {
          setMetrics(response.data.metrics);
        }
      }
    } catch (error) {
      addLog(`✗ Error loading model: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const saveModelAs = async (modelName) => {
    setIsSaving(true);
    addLog(`Saving model as ${modelName}...`, 'info');

    try {
      const response = await axios.post(`${API_BASE}/save_model`, {
        model_name: modelName
      });

      if (response.data.success) {
        addLog(`✓ Model saved as ${modelName}: ${response.data.path}`, 'success');
        // Refresh available models
        fetchAvailableModels();
      }
    } catch (error) {
      addLog(`✗ Error saving model: ${error.message}`, 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/list_models`);
      setAvailableModels(response.data.models);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchTrainingSessions = async () => {
    try {
      const response = await axios.get(`${API_BASE}/training_sessions`);
      setTrainingSessions(response.data.sessions);
    } catch (error) {
      console.error('Error fetching training sessions:', error);
    }
  };

  const generateFromModel = async (modelName, setImage) => {
    try {
      const response = await axios.post(`${API_BASE}/generate_from_model`, {
        model_name: modelName,
        num_images: 16
      });

      if (response.data.success) {
        setImage(response.data.image);
        return true;
      }
    } catch (error) {
      console.error(`Error generating from ${modelName}:`, error);
      return false;
    }
  };

  const loadComparisonData = async () => {
    setIsLoadingComparison(true);
    addLog('Loading comparison data...', 'info');

    try {
      // Fetch available models and training sessions
      await fetchAvailableModels();
      await fetchTrainingSessions();

      // Generate images from both models if available
      const models = await axios.get(`${API_BASE}/list_models`);
      const modelNames = models.data.models.map(m => m.name);

      if (modelNames.includes('mnist')) {
        await generateFromModel('mnist', setMnistImage);
      }
      if (modelNames.includes('fashion_mnist')) {
        await generateFromModel('fashion_mnist', setFashionImage);
      }

      setShowComparison(true);
      addLog('✓ Comparison data loaded', 'success');
    } catch (error) {
      addLog(`✗ Error loading comparison: ${error.message}`, 'error');
    } finally {
      setIsLoadingComparison(false);
    }
  };

  // Prepare data for charts
  const lossData = metrics.g_losses.map((g_loss, idx) => ({
    epoch: idx,
    'Generator Loss': g_loss,
    'Discriminator Loss': metrics.d_losses[idx]
  }));

  const scoreData = metrics.real_scores.map((real_score, idx) => ({
    epoch: idx,
    'D(real)': real_score,
    'D(fake)': metrics.fake_scores[idx]
  }));

  return (
    <div className="app">
      <header className="header">
        <h1>🎨 DCGAN Interactive Demo</h1>
        <p className="subtitle">Deep Convolutional Generative Adversarial Network</p>
        <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
          {connected ? '● Connected' : '○ Disconnected'}
        </div>
      </header>

      <div className="main-container">
        {/* Control Panel */}
        <div className="panel control-panel">
          <h2>Training Controls</h2>

          <div className="form-group">
            <label>Dataset:</label>
            <select
              value={config.dataset}
              onChange={(e) => setConfig({ ...config, dataset: e.target.value })}
              disabled={status.is_training}
            >
              <option value="mnist">MNIST (Digits)</option>
              <option value="fashion_mnist">Fashion MNIST (Clothing)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Device:</label>
            <select
              value={config.device}
              onChange={(e) => setConfig({ ...config, device: e.target.value })}
              disabled={status.is_training}
            >
              <option value="mps">GPU (Metal/MPS - Mac)</option>
              <option value="cuda">GPU (CUDA - NVIDIA)</option>
              <option value="cpu">CPU</option>
            </select>
            <small style={{display: 'block', marginTop: '4px', color: '#666'}}>
              {config.device === 'mps' ? '⚡ Fast training (~1-3 min/epoch)' :
               config.device === 'cuda' ? '⚡ Very fast training (~30-60 sec/epoch)' :
               '🐌 Slow training (~20-50 min/epoch)'}
            </small>
          </div>

          <div className="form-group">
            <label>Epochs:</label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
              disabled={status.is_training}
              min="1"
              max="100"
            />
          </div>

          <div className="form-group">
            <label>Batch Size:</label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
              disabled={status.is_training}
              min="16"
              max="256"
              step="16"
            />
          </div>

          <div className="form-group">
            <label>Learning Rate:</label>
            <input
              type="number"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
              disabled={status.is_training}
              min="0.0001"
              max="0.01"
              step="0.0001"
            />
          </div>

          <div className="button-group">
            <button
              className="btn btn-primary"
              onClick={startTraining}
              disabled={status.is_training || isStarting}
            >
              {isStarting ? '⏳ Starting...' : status.is_training ? '🔄 Training...' : '▶️ Start Training'}
            </button>
            <button
              className="btn btn-danger"
              onClick={stopTraining}
              disabled={!status.is_training}
            >
              ⏹️ Stop Training
            </button>
          </div>

          <div className="button-group" style={{marginTop: '10px'}}>
            <button
              className="btn btn-secondary"
              onClick={saveModel}
              disabled={isSaving || status.is_training}
            >
              {isSaving ? '⏳ Saving...' : '💾 Save Model'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={loadModel}
              disabled={isLoading || status.is_training}
            >
              {isLoading ? '⏳ Loading...' : '📂 Load Model'}
            </button>
          </div>

          <div className="status-info">
            <p>
              <strong>Status:</strong>{' '}
              {isStarting ? '⏳ Starting...' :
               status.is_training ? '🔄 Training' :
               '✓ Idle'}
            </p>
            <p>
              <strong>Device:</strong>{' '}
              {status.device === 'mps' ? '⚡ GPU (Metal)' :
               status.device === 'cuda' || (status.device && status.device.includes('cuda')) ? '⚡ GPU (CUDA)' :
               '🖥️ CPU'}
            </p>
            <p><strong>Current Epoch:</strong> {status.current_epoch} / {config.epochs}</p>
            {status.is_training && (
              <div className="progress-bar-container">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${(status.current_epoch / config.epochs) * 100}%` }}
                />
              </div>
            )}
          </div>
        </div>

        {/* Image Generation Panel */}
        <div className="panel generation-panel">
          <h2>Generate Images</h2>
          <p>Generate synthetic images using the trained generator</p>

          <div className="button-group">
            <button
              className="btn btn-success"
              onClick={() => generateImages(16)}
              disabled={isGenerating}
            >
              {isGenerating ? '⏳ Generating...' : '🎨 Generate 16 Images'}
            </button>
            <button
              className="btn btn-success"
              onClick={() => generateImages(64)}
              disabled={isGenerating}
            >
              {isGenerating ? '⏳ Generating...' : '🎨 Generate 64 Images'}
            </button>
          </div>

          {generatedImage && (
            <div className="image-container">
              <h3>Generated Images</h3>
              <img
                src={`data:image/png;base64,${generatedImage}`}
                alt="Generated samples"
                className="generated-image"
              />
            </div>
          )}
        </div>

        {/* Training Progress Panel */}
        <div className="panel progress-panel">
          <h2>Training Progress</h2>

          {currentImage ? (
            <div className="image-container">
              <h3>Latest Generated Samples (Epoch {status.current_epoch})</h3>
              <img
                src={`data:image/png;base64,${currentImage}`}
                alt="Training samples"
                className="training-image"
              />
            </div>
          ) : (
            <div className="placeholder">
              <p>Training samples will appear here during training</p>
              <p className="hint">Click "Start Training" to begin</p>
            </div>
          )}
        </div>

        {/* Loss Charts */}
        <div className="panel chart-panel">
          <h2>Loss Curves</h2>
          {lossData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="Generator Loss" stroke="#8b5cf6" strokeWidth={2} />
                <Line type="monotone" dataKey="Discriminator Loss" stroke="#ec4899" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="placeholder">
              <p>Loss curves will appear here during training</p>
            </div>
          )}
        </div>

        {/* Discriminator Scores */}
        <div className="panel chart-panel">
          <h2>Discriminator Scores</h2>
          {scoreData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={scoreData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[0, 1]} label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="D(real)" stroke="#10b981" strokeWidth={2} />
                <Line type="monotone" dataKey="D(fake)" stroke="#f59e0b" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="placeholder">
              <p>Discriminator scores will appear here during training</p>
              <p className="hint">D(real) should stay near 1, D(fake) should increase toward 0.5</p>
            </div>
          )}
        </div>

        {/* Training Logs */}
        <div className="panel logs-panel">
          <h2>Training Logs</h2>
          <div className="logs-container">
            {logs.length === 0 ? (
              <p className="placeholder">Logs will appear here...</p>
            ) : (
              <>
                {logs.map((log, idx) => (
                  <div key={idx} className={`log-entry log-${log.type}`}>
                    <span className="log-time">{log.timestamp}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </>
            )}
          </div>
        </div>

        {/* Architecture Info */}
        <div className="panel info-panel">
          <h2>DCGAN Architecture</h2>

          <div className="architecture-grid">
            <div className="arch-block">
              <h3>Generator (G)</h3>
              <div className="arch-details">
                <p><strong>Input:</strong> z ∈ ℝ<sup>100</sup> (noise vector)</p>
                <p><strong>Architecture:</strong></p>
                <ul>
                  <li>ConvTranspose2d: 100 → 512 (4×4)</li>
                  <li>ConvTranspose2d: 512 → 256 (8×8)</li>
                  <li>ConvTranspose2d: 256 → 128 (16×16)</li>
                  <li>ConvTranspose2d: 128 → 64 (32×32)</li>
                  <li>ConvTranspose2d: 64 → 3 (64×64)</li>
                </ul>
                <p><strong>Output:</strong> 3×64×64 RGB image</p>
              </div>
            </div>

            <div className="arch-block">
              <h3>Discriminator (D)</h3>
              <div className="arch-details">
                <p><strong>Input:</strong> 3×64×64 RGB image</p>
                <p><strong>Architecture:</strong></p>
                <ul>
                  <li>Conv2d: 3 → 64 (32×32)</li>
                  <li>Conv2d: 64 → 128 (16×16)</li>
                  <li>Conv2d: 128 → 256 (8×8)</li>
                  <li>Conv2d: 256 → 512 (4×4)</li>
                  <li>Conv2d: 512 → 1 (1×1)</li>
                </ul>
                <p><strong>Output:</strong> Probability ∈ [0, 1]</p>
              </div>
            </div>
          </div>

          <div className="training-info">
            <h3>Training Algorithm</h3>
            <ol>
              <li><strong>Update D:</strong> Maximize log D(x) + log(1 - D(G(z)))</li>
              <li><strong>Update G:</strong> Maximize log D(G(z)) (non-saturating)</li>
              <li><strong>Optimizer:</strong> Adam (lr=0.0002, β₁=0.5)</li>
              <li><strong>Loss:</strong> Binary Cross-Entropy</li>
            </ol>
          </div>
        </div>

        {/* Comparison Dashboard */}
        <div className="panel comparison-panel" style={{ gridColumn: '1 / -1' }}>
          <h2>MNIST vs Fashion-MNIST Comparison</h2>

          <div className="button-group" style={{ marginBottom: '20px' }}>
            <button
              className="btn btn-primary"
              onClick={loadComparisonData}
              disabled={isLoadingComparison || status.is_training}
            >
              {isLoadingComparison ? '⏳ Loading...' : '📊 Load Comparison Data'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => generateFromModel('mnist', setMnistImage)}
              disabled={status.is_training}
            >
              🔢 Generate MNIST
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => generateFromModel('fashion_mnist', setFashionImage)}
              disabled={status.is_training}
            >
              👕 Generate Fashion
            </button>
          </div>

          {/* Side-by-side Image Comparison */}
          <div className="comparison-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div className="comparison-card" style={{ background: '#f8f9ff', padding: '15px', borderRadius: '10px', border: '2px solid #667eea' }}>
              <h3 style={{ color: '#667eea', marginBottom: '10px' }}>MNIST (Digits)</h3>
              {mnistImage ? (
                <img
                  src={`data:image/png;base64,${mnistImage}`}
                  alt="MNIST generated"
                  style={{ width: '100%', borderRadius: '8px' }}
                />
              ) : (
                <div style={{ padding: '40px', textAlign: 'center', color: '#666', background: '#eee', borderRadius: '8px' }}>
                  <p>No MNIST model loaded</p>
                  <small>Train and save an MNIST model first</small>
                </div>
              )}
            </div>

            <div className="comparison-card" style={{ background: '#fff5f5', padding: '15px', borderRadius: '10px', border: '2px solid #e74c3c' }}>
              <h3 style={{ color: '#e74c3c', marginBottom: '10px' }}>Fashion-MNIST (Clothing)</h3>
              {fashionImage ? (
                <img
                  src={`data:image/png;base64,${fashionImage}`}
                  alt="Fashion-MNIST generated"
                  style={{ width: '100%', borderRadius: '8px' }}
                />
              ) : (
                <div style={{ padding: '40px', textAlign: 'center', color: '#666', background: '#eee', borderRadius: '8px' }}>
                  <p>No Fashion-MNIST model loaded</p>
                  <small>Train and save a Fashion-MNIST model first</small>
                </div>
              )}
            </div>
          </div>

          {/* Training Sessions Metrics Comparison */}
          {(trainingSessions.mnist || trainingSessions.fashion_mnist) && (
            <div className="metrics-comparison" style={{ marginTop: '20px' }}>
              <h3 style={{ marginBottom: '15px' }}>Training Metrics Comparison</h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '15px' }}>
                <thead>
                  <tr style={{ background: '#667eea', color: 'white' }}>
                    <th style={{ padding: '12px', textAlign: 'left', border: '1px solid #ddd' }}>Metric</th>
                    <th style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>MNIST</th>
                    <th style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>Fashion-MNIST</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ background: '#f8f9ff' }}>
                    <td style={{ padding: '12px', border: '1px solid #ddd' }}>Training Time</td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.mnist?.training_time_formatted || '-'}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.fashion_mnist?.training_time_formatted || '-'}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '12px', border: '1px solid #ddd' }}>Epochs</td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.mnist?.epochs || '-'}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.fashion_mnist?.epochs || '-'}
                    </td>
                  </tr>
                  <tr style={{ background: '#f8f9ff' }}>
                    <td style={{ padding: '12px', border: '1px solid #ddd' }}>Final Generator Loss</td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.mnist?.final_g_loss?.toFixed(4) || '-'}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.fashion_mnist?.final_g_loss?.toFixed(4) || '-'}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '12px', border: '1px solid #ddd' }}>Final Discriminator Loss</td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.mnist?.final_d_loss?.toFixed(4) || '-'}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.fashion_mnist?.final_d_loss?.toFixed(4) || '-'}
                    </td>
                  </tr>
                  <tr style={{ background: '#f8f9ff' }}>
                    <td style={{ padding: '12px', border: '1px solid #ddd' }}>D(real) Score</td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.mnist?.final_real_score?.toFixed(4) || '-'}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.fashion_mnist?.final_real_score?.toFixed(4) || '-'}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '12px', border: '1px solid #ddd' }}>D(fake) Score</td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.mnist?.final_fake_score?.toFixed(4) || '-'}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.fashion_mnist?.final_fake_score?.toFixed(4) || '-'}
                    </td>
                  </tr>
                  <tr style={{ background: '#f8f9ff' }}>
                    <td style={{ padding: '12px', border: '1px solid #ddd' }}>Device</td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.mnist?.device || '-'}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center', border: '1px solid #ddd' }}>
                      {trainingSessions.fashion_mnist?.device || '-'}
                    </td>
                  </tr>
                </tbody>
              </table>

              {/* Comparison Loss Curves */}
              {(trainingSessions.mnist?.metrics?.g_losses?.length > 0 || trainingSessions.fashion_mnist?.metrics?.g_losses?.length > 0) && (
                <div style={{ marginTop: '20px' }}>
                  <h3 style={{ marginBottom: '15px' }}>Loss Curves Comparison</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                    {/* MNIST Loss Curve */}
                    <div style={{ background: '#f8f9ff', padding: '15px', borderRadius: '10px' }}>
                      <h4 style={{ color: '#667eea', marginBottom: '10px' }}>MNIST Generator Loss</h4>
                      {trainingSessions.mnist?.metrics?.g_losses?.length > 0 ? (
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={trainingSessions.mnist.metrics.g_losses.map((loss, idx) => ({ epoch: idx, loss }))}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="epoch" />
                            <YAxis />
                            <Tooltip />
                            <Line type="monotone" dataKey="loss" stroke="#667eea" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <p style={{ color: '#666', textAlign: 'center' }}>No data</p>
                      )}
                    </div>

                    {/* Fashion-MNIST Loss Curve */}
                    <div style={{ background: '#fff5f5', padding: '15px', borderRadius: '10px' }}>
                      <h4 style={{ color: '#e74c3c', marginBottom: '10px' }}>Fashion-MNIST Generator Loss</h4>
                      {trainingSessions.fashion_mnist?.metrics?.g_losses?.length > 0 ? (
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={trainingSessions.fashion_mnist.metrics.g_losses.map((loss, idx) => ({ epoch: idx, loss }))}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="epoch" />
                            <YAxis />
                            <Tooltip />
                            <Line type="monotone" dataKey="loss" stroke="#e74c3c" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <p style={{ color: '#666', textAlign: 'center' }}>No data</p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Available Models */}
          {availableModels.length > 0 && (
            <div className="available-models" style={{ marginTop: '20px' }}>
              <h3 style={{ marginBottom: '10px' }}>Saved Models</h3>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                {availableModels.map((model) => (
                  <div
                    key={model.name}
                    style={{
                      padding: '10px 15px',
                      background: model.name === 'mnist' ? '#e8f5e9' : model.name === 'fashion_mnist' ? '#fff3e0' : '#f5f5f5',
                      borderRadius: '8px',
                      border: '1px solid #ddd'
                    }}
                  >
                    <strong>{model.name}</strong>
                    <div style={{ fontSize: '0.85em', color: '#666' }}>
                      {model.size_mb} MB | {model.modified}
                    </div>
                    <button
                      className="btn btn-sm"
                      style={{ marginTop: '5px', padding: '4px 8px', fontSize: '0.85em' }}
                      onClick={() => loadModel(model.name)}
                      disabled={isLoading || status.is_training}
                    >
                      Load
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model Switching */}
          <div className="model-switching" style={{ marginTop: '20px', padding: '15px', background: '#f0f4ff', borderRadius: '10px' }}>
            <h3 style={{ marginBottom: '10px' }}>Quick Model Actions</h3>
            <div className="button-group">
              <button
                className="btn btn-secondary"
                onClick={() => loadModel('mnist')}
                disabled={isLoading || status.is_training}
              >
                📂 Load MNIST Model
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => loadModel('fashion_mnist')}
                disabled={isLoading || status.is_training}
              >
                📂 Load Fashion Model
              </button>
              <button
                className="btn btn-success"
                onClick={() => saveModelAs('mnist')}
                disabled={isSaving || status.is_training}
              >
                💾 Save as MNIST
              </button>
              <button
                className="btn btn-success"
                onClick={() => saveModelAs('fashion_mnist')}
                disabled={isSaving || status.is_training}
              >
                💾 Save as Fashion
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
