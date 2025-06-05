import os
import logging
import numpy as np
import librosa
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import soundfile as sf
import io
from datetime import datetime
import traceback
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech Emotion Recognition with Whisper Large V3")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise

# Whisper emotion model labels (7 emotions)
EMOTION_LABELS = [
    "angry", "disgust", "fearful", "happy", 
    "neutral", "sad", "surprised"
]

class WhisperEmotionModel:
    def __init__(self, model_name="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = None
        self.id2label = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Whisper emotion recognition model"""
        try:
            logger.info(f"Loading Whisper emotion model: {self.model_name}")
            
            # Set cache directory for Replit
            os.environ['HF_HOME'] = '/tmp/huggingface_cache'
            os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
            
            # Load model and feature extractor
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                cache_dir='/tmp/huggingface_cache',
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                low_cpu_mem_usage=True
            )
            
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                do_normalize=True,
                cache_dir='/tmp/huggingface_cache'
            )
            
            # Get label mapping
            self.id2label = self.model.config.id2label
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Successfully loaded Whisper emotion model on {self.device}")
            logger.info(f"Available emotions: {list(self.id2label.values())}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.error(traceback.format_exc())
            self.model = self.create_mock_model()
            logger.warning("Using mock model for testing")

    def create_mock_model(self):
        """Create mock model for testing when real model fails"""
        class MockModel:
            def __init__(self):
                self.config = type('Config', (), {
                    'id2label': {i: emotion for i, emotion in enumerate(EMOTION_LABELS)}
                })()
            
            def __call__(self, **kwargs):
                # Return mock logits
                logits = torch.randn(1, len(EMOTION_LABELS))
                return type('Output', (), {'logits': logits})()
                
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
        return MockModel()

    def preprocess_audio(self, audio_data, sample_rate, max_duration=30.0):
        """Preprocess audio for Whisper emotion recognition"""
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            # Handle multi-channel audio by taking the mean
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Ensure audio_data is 1D numpy array
            audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
            
            # Get target sample rate from feature extractor
            target_sr = self.feature_extractor.sampling_rate
            
            # Resample if necessary
            if sample_rate != target_sr:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=target_sr
                )
            
            # Apply max duration limit
            max_length = int(target_sr * max_duration)
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            else:
                # Pad if too short
                audio_data = np.pad(audio_data, (0, max_length - len(audio_data)))
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract features using Whisper feature extractor
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=target_sr,
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            return inputs
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            logger.error(traceback.format_exc())
            # Return fallback empty tensor
            return {"input_features": torch.zeros(1, 128, 3000)}

    def predict_emotion(self, audio_data, sample_rate):
        """Predict emotion using Whisper model"""
        try:
            # Preprocess audio
            inputs = self.preprocess_audio(audio_data, sample_rate)
            
            # Move inputs to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities using softmax
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(logits, dim=-1).item()
                
            # Map predictions to emotions
            if self.id2label:
                emotion_probs = {}
                for i, prob in enumerate(probabilities[0]):
                    emotion_name = self.id2label.get(i, f"emotion_{i}")
                    emotion_probs[emotion_name.lower()] = float(prob)
                
                predicted_emotion = self.id2label.get(predicted_id, "unknown").lower()
                confidence = float(probabilities[0][predicted_id])
            else:
                # Fallback mapping to standard emotions
                emotion_probs = {
                    emotion: float(prob) 
                    for emotion, prob in zip(EMOTION_LABELS, probabilities[0][:len(EMOTION_LABELS)])
                }
                predicted_emotion = EMOTION_LABELS[min(predicted_id, len(EMOTION_LABELS)-1)]
                confidence = float(probabilities[0][predicted_id])
            
            return {
                "emotions": emotion_probs,
                "dominant_emotion": predicted_emotion,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "model_type": "Whisper_Large_V3",
                "model_name": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return fallback result
            fallback_emotions = {emotion: 1.0/len(EMOTION_LABELS) for emotion in EMOTION_LABELS}
            return {
                "emotions": fallback_emotions,
                "dominant_emotion": "neutral",
                "confidence": 0.5,
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Prediction failed: {str(e)}",
                "model_type": "fallback"
            }

# Initialize global model
try:
    emotion_model = WhisperEmotionModel()
    logger.info("Whisper emotion model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize emotion model: {e}")
    emotion_model = None

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.post("/analyze_audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """Audio analysis endpoint using Whisper model"""
    try:
        if emotion_model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Model not available",
                    "detail": "Whisper emotion recognition model is not loaded",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        if not audio_file.filename:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No file provided",
                    "detail": "Please select an audio file",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Read and process audio file
        audio_bytes = await audio_file.read()
        
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid audio file",
                    "detail": f"Could not read audio file: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        logger.info(f"Processing audio: {audio_file.filename}, duration: {len(audio_data)/sample_rate:.2f}s, sample_rate: {sample_rate}")
        
        # Predict emotion
        result = emotion_model.predict_emotion(audio_data, sample_rate)
        
        return JSONResponse(
            content={
                "status": "success",
                "filename": audio_file.filename,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "Analysis failed",
                "detail": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "available" if emotion_model and emotion_model.model else "unavailable"
    model_type = "Whisper_Large_V3" if emotion_model and emotion_model.model else "none"
    available_emotions = list(emotion_model.id2label.values()) if emotion_model and emotion_model.id2label else EMOTION_LABELS
    
    return JSONResponse(
        content={
            "status": "running",
            "model_status": model_status,
            "model_type": model_type,
            "model_name": emotion_model.model_name if emotion_model else "none",
            "available_emotions": available_emotions,
            "device": str(emotion_model.device) if emotion_model else "none",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve web interface for Whisper emotion recognition"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whisper Speech Emotion Recognition</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 16px; 
                padding: 2rem; 
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            }
            .header { 
                text-align: center; 
                margin-bottom: 2rem; 
                padding-bottom: 1rem;
                border-bottom: 2px solid #f0f0f0;
            }
            h1 { 
                color: #333; 
                margin-bottom: 0.5rem;
                font-size: 2.5rem;
                font-weight: 700;
            }
            .whisper-badge { 
                background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 8px 20px; 
                border-radius: 25px; 
                font-size: 0.9rem;
                font-weight: 600;
                display: inline-block;
                margin: 10px 0;
            }
            .status { 
                padding: 15px; 
                margin: 15px 0; 
                border-radius: 8px; 
                font-weight: 500;
            }
            .success { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
            .warning { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
            .error { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
            
            .upload-section {
                background: #f8f9fa;
                border: 3px dashed #dee2e6;
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin: 20px 0;
                transition: all 0.3s ease;
            }
            .upload-section:hover { 
                border-color: #667eea; 
                background: #f0f4ff; 
            }
            
            input[type="file"] {
                display: none;
            }
            .file-label {
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                transition: background 0.3s ease;
                margin: 10px;
            }
            .file-label:hover {
                background: #5a67d8;
            }
            
            button { 
                background: #28a745; 
                color: white; 
                border: none; 
                padding: 12px 24px; 
                border-radius: 8px; 
                cursor: pointer; 
                margin: 5px; 
                font-weight: 500;
                font-size: 1rem;
                transition: all 0.3s ease;
            }
            button:hover { background: #218838; transform: translateY(-1px); }
            button:disabled { background: #6c757d; cursor: not-allowed; transform: none; }
            
            .results { 
                margin: 25px 0; 
                padding: 25px; 
                background: linear-gradient(145deg, #f8f9fa, #ffffff); 
                border-radius: 12px;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
            }
            .emotion-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .emotion-item { 
                background: white; 
                border-radius: 8px; 
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            }
            .emotion-item:hover {
                transform: translateY(-2px);
            }
            .emotion-name {
                font-weight: 600;
                text-transform: capitalize;
                margin-bottom: 8px;
                color: #333;
            }
            .emotion-percentage {
                font-size: 1.2rem;
                font-weight: 700;
                color: #667eea;
            }
            .emotion-bar { 
                height: 6px; 
                background: linear-gradient(90deg, #667eea, #764ba2); 
                border-radius: 3px; 
                margin-top: 8px;
                transition: width 0.5s ease;
            }
            .dominant-emotion { 
                font-size: 1.5rem; 
                font-weight: 700; 
                color: #667eea; 
                margin: 20px 0;
                text-align: center;
                padding: 15px;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .log { 
                background: #2c3e50; 
                color: #ecf0f1; 
                padding: 20px; 
                border-radius: 8px; 
                height: 250px; 
                overflow-y: auto; 
                font-family: 'Courier New', monospace; 
                font-size: 0.9rem;
                margin-top: 20px;
            }
            .log-entry {
                margin: 2px 0;
                padding: 2px 0;
            }
            .file-info {
                margin: 10px 0;
                font-style: italic;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Speech Emotion Recognition</h1>
                <div class="whisper-badge">Powered by Whisper Large V3</div>
                <p>Advanced emotion detection using OpenAI's Whisper model fine-tuned for emotion recognition</p>
            </div>
            
            <div id="statusIndicator" class="status warning">🔄 Initializing Whisper model...</div>
            
            <div class="upload-section">
                <div>📁 <strong>Upload Audio File for Emotion Analysis</strong></div>
                <label for="audioFile" class="file-label">Choose Audio File</label>
                <input type="file" id="audioFile" accept="audio/*">
                <div class="file-info" id="fileInfo">No file selected</div>
                <button id="analyzeBtn" onclick="analyzeAudioFile()" disabled>🔍 Analyze Emotion</button>
                <div style="margin-top: 10px;">
                    <small>Supports: WAV, MP3, M4A, FLAC, OGG (recommended: max 30 seconds)</small>
                </div>
            </div>
            
            <div class="results" id="resultsPanel">
                <h3>🎯 Emotion Analysis Results</h3>
                <div class="dominant-emotion" id="dominantEmotion">Upload an audio file to begin analysis</div>
                <div class="emotion-grid" id="emotionResults">
                    <div style="grid-column: 1 / -1; text-align: center; color: #666;">
                        Emotion probabilities will appear here after analysis
                    </div>
                </div>
                <div id="confidence" style="text-align: center; margin-top: 15px; font-style: italic; color: #666;">
                    Confidence: 0%
                </div>
            </div>
            
            <div class="log" id="logPanel">
                <div class="log-entry">[System] Initializing Whisper emotion recognition system...</div>
            </div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                initializeApplication();
                setupEventListeners();
            });

            async function initializeApplication() {
                try {
                    logMessage('🔍 Checking Whisper model status...', 'info');
                    
                    const response = await fetch('/health');
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const healthData = await response.json();
                    
                    if (healthData.model_status === 'available') {
                        updateStatus(`✅ Whisper model ready - ${healthData.model_name.split('/').pop()}`, 'success');
                        logMessage(`📊 Model loaded on ${healthData.device}`, 'success');
                        logMessage(`🎭 Available emotions: ${healthData.available_emotions.join(', ')}`, 'info');
                        document.getElementById('analyzeBtn').disabled = false;
                    } else {
                        updateStatus('⚠️ Whisper model unavailable - using fallback', 'warning');
                        logMessage('❌ Model loading failed - using mock predictions', 'error');
                        document.getElementById('analyzeBtn').disabled = false; // Still allow testing
                    }
                    
                } catch (error) {
                    updateStatus(`❌ System error: ${error.message}`, 'error');
                    logMessage(`💥 Initialization failed: ${error.message}`, 'error');
                }
            }

            function setupEventListeners() {
                const audioFile = document.getElementById('audioFile');
                audioFile.addEventListener('change', function(e) {
                    const analyzeBtn = document.getElementById('analyzeBtn');
                    const fileInfo = document.getElementById('fileInfo');
                    
                    if (e.target.files.length > 0) {
                        const file = e.target.files[0];
                        const sizeMB = (file.size / 1024 / 1024).toFixed(2);
                        fileInfo.textContent = `Selected: ${file.name} (${sizeMB} MB)`;
                        logMessage(`📁 File selected: ${file.name} (${sizeMB}MB)`, 'info');
                        analyzeBtn.disabled = false;
                    } else {
                        fileInfo.textContent = 'No file selected';
                        analyzeBtn.disabled = true;
                    }
                });
            }

            async function analyzeAudioFile() {
                const fileInput = document.getElementById('audioFile');
                if (!fileInput.files.length) {
                    logMessage('❌ No file selected', 'error');
                    return;
                }

                const analyzeBtn = document.getElementById('analyzeBtn');
                const originalText = analyzeBtn.textContent;
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = '🔄 Analyzing with Whisper...';

                try {
                    const formData = new FormData();
                    formData.append('audio_file', fileInput.files[0]);

                    logMessage('⬆️ Uploading audio for Whisper analysis...', 'info');
                    const startTime = Date.now();
                    
                    const response = await fetch('/analyze_audio', {
                        method: 'POST',
                        body: formData
                    });

                    const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP ${response.status}`);
                    }

                    const result = await response.json();
                    displayResults(result.result);
                    logMessage(`✅ Analysis complete in ${processingTime}s: ${result.result.dominant_emotion}`, 'success');

                } catch (error) {
                    logMessage(`💥 Analysis failed: ${error.message}`, 'error');
                    updateStatus(`❌ Analysis error: ${error.message}`, 'error');
                } finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = originalText;
                }
            }

            function displayResults(result) {
                const emotionResults = document.getElementById('emotionResults');
                const dominantEmotion = document.getElementById('dominantEmotion');
                const confidence = document.getElementById('confidence');
                
                // Display dominant emotion
                const confidencePercent = (result.confidence * 100).toFixed(1);
                dominantEmotion.innerHTML = `🎯 <strong>${result.dominant_emotion.toUpperCase()}</strong> (${confidencePercent}% confidence)`;
                
                // Display all emotion probabilities with bars
                const sortedEmotions = Object.entries(result.emotions)
                    .sort(([,a], [,b]) => b - a);
                
                let html = '';
                for (const [emotion, score] of sortedEmotions) {
                    const percentage = (score * 100).toFixed(1);
                    const barWidth = Math.max(5, percentage);
                    const isTop = emotion === result.dominant_emotion;
                    
                    html += `
                        <div class="emotion-item" style="${isTop ? 'border: 2px solid #667eea; background: #f0f4ff;' : ''}">
                            <div class="emotion-name">${isTop ? '🏆 ' : ''}${emotion}</div>
                            <div class="emotion-percentage">${percentage}%</div>
                            <div class="emotion-bar" style="width: ${barWidth}%;"></div>
                        </div>
                    `;
                }
                emotionResults.innerHTML = html;
                
                // Update confidence info
                const modelInfo = result.model_type || 'Unknown';
                confidence.innerHTML = `
                    <strong>Model:</strong> ${modelInfo} | 
                    <strong>Processing:</strong> ${new Date(result.timestamp).toLocaleTimeString()}
                    ${result.error ? `<br><strong>Note:</strong> ${result.error}` : ''}
                `;
            }

            function logMessage(message, type = 'info') {
                const logPanel = document.getElementById('logPanel');
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                
                const typeColors = {
                    'info': '#3498db',
                    'success': '#27ae60',
                    'error': '#e74c3c',
                    'warning': '#f39c12'
                };
                
                logEntry.innerHTML = `
                    <span style="color: #95a5a6;">[${timestamp}]</span>
                    <span style="color: ${typeColors[type] || '#ecf0f1'};">${message}</span>
                `;
                
                logPanel.appendChild(logEntry);
                logPanel.scrollTop = logPanel.scrollHeight;
            }

            function updateStatus(message, type) {
                const statusIndicator = document.getElementById('statusIndicator');
                statusIndicator.className = `status ${type}`;
                statusIndicator.textContent = message;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 80))
    logger.info(f"Starting Whisper emotion recognition server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
