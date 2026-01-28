# Real-Time Recording & Streaming - Implementation Guide

## Feature: B1 & B2 - Real-Time Audio/Video Recording

**Priority:** Medium  
**Estimated Time:** 2-3 weeks  
**Dependencies:** Basic detection working

---

## 1. Requirements

### Functional Requirements
- [ ] Record audio from microphone in real-time
- [ ] Record video from webcam in real-time
- [ ] Stream data to backend for analysis
- [ ] Provide live feedback during recording
- [ ] Support recording sessions up to 30 minutes
- [ ] Allow pause/resume functionality

### Non-Functional Requirements
- [ ] Latency <500ms for audio feedback
- [ ] Latency <2s for video feedback
- [ ] Support 10+ concurrent recording sessions
- [ ] Bandwidth: <1 Mbps per audio stream, <3 Mbps per video stream
- [ ] Works on Chrome, Firefox, Safari (WebRTC support)

### User Experience Requirements
- [ ] Clear permission prompts for mic/camera
- [ ] Real-time confidence meter
- [ ] Visual indicators during recording
- [ ] Graceful handling of permission denials
- [ ] Mobile-friendly interface

---

## 2. Architecture

### Component Overview

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Browser    │◄───────►│   Backend    │◄───────►│  ML Model    │
│ (WebRTC/JS)  │  WSS    │  (FastAPI)   │         │   (ONNX)     │
└──────────────┘         └──────────────┘         └──────────────┘
       │                        │                         │
       │ Audio/Video            │ Chunks                  │ Predictions
       │ Chunks                 │                         │
       ▼                        ▼                         ▼
  MediaRecorder          WebSocket Handler        Streaming Detector
```

### Technology Stack

**Frontend:**
- WebRTC MediaStream API (audio/video capture)
- MediaRecorder API (chunking)
- WebSocket (streaming to backend)
- React/Vue for UI (optional)

**Backend:**
- FastAPI WebSocket support
- Async processing (asyncio)
- Chunked audio/video processing
- Sliding window analysis

**ML:**
- Streaming inference (batch processing)
- Temporal buffer (3-5 seconds context)
- Real-time aggregation

### Data Flow

1. **Capture**: Browser → MediaRecorder → Audio/Video chunks
2. **Stream**: WebSocket → Backend (binary data)
3. **Buffer**: Accumulate 3-5 seconds of data
4. **Analyze**: ML model inference on buffer
5. **Respond**: Verdict + confidence → WebSocket → UI update
6. **Loop**: Continue until recording stops

---

## 3. Implementation Phases

### Phase 1: WebSocket Infrastructure (Week 1, Days 1-3)

**Tasks:**
- [ ] Add WebSocket endpoint to FastAPI
- [ ] Implement connection management (connect/disconnect/error)
- [ ] Create message protocol (audio/video data, metadata)
- [ ] Add session tracking and cleanup

**Backend Code:**

```python
# services/api/app/streaming.py (NEW)

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio
import uuid

class StreamingSession:
    def __init__(self, session_id: str, media_type: str):
        self.session_id = session_id
        self.media_type = media_type  # "audio" or "video"
        self.buffer = bytearray()
        self.analysis_id = None
        self.created_at = datetime.utcnow()
    
    async def add_chunk(self, data: bytes):
        self.buffer.extend(data)
        
        # Process when buffer reaches threshold (e.g., 3 seconds)
        if len(self.buffer) >= self.get_buffer_threshold():
            await self.process_buffer()
    
    async def process_buffer(self):
        # Extract features and run inference
        pass

class ConnectionManager:
    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_sessions[session_id] = websocket
    
    async def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def send_result(self, session_id: str, verdict: dict):
        websocket = self.active_sessions.get(session_id)
        if websocket:
            await websocket.send_json(verdict)

manager = ConnectionManager()

# services/api/app/main.py (UPDATE)

@app.websocket("/v1/stream/audio")
async def websocket_audio_stream(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    session = StreamingSession(session_id, "audio")
    
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive binary audio chunk
            data = await websocket.receive_bytes()
            
            await session.add_chunk(data)
            
            # Send real-time feedback
            if session.has_result():
                result = session.get_latest_result()
                await manager.send_result(session_id, result)
    
    except WebSocketDisconnect:
        await manager.disconnect(session_id)
        await session.finalize()
```

**Testing:**
```python
# tests/test_streaming.py

async def test_websocket_connection():
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.websocket_connect("/v1/stream/audio") as ws:
            # Send dummy audio chunk
            await ws.send_bytes(b"dummy_audio_data")
            
            # Should receive acknowledgment or result
            response = await ws.receive_json()
            assert "session_id" in response

async def test_streaming_disconnect():
    # Test graceful disconnect handling
    pass

async def test_concurrent_streams():
    # Test multiple simultaneous streaming sessions
    pass
```

### Phase 2: Frontend Audio Recording (Week 1, Days 4-5)

**Tasks:**
- [ ] Implement MediaRecorder for audio
- [ ] Add microphone permission handling
- [ ] Create WebSocket client
- [ ] Display real-time confidence meter

**Frontend Code:**

```javascript
// frontend/src/components/AudioRecorder.js

class AudioRecorder {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.mediaRecorder = null;
        this.stream = null;
    }
    
    async start() {
        try {
            // Request microphone permission
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });
            
            // Setup MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            // Send chunks every 1 second
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(event.data);
                }
            };
            
            // Setup WebSocket
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.mediaRecorder.start(1000); // Chunk every 1s
            };
            
            this.ws.onmessage = (event) => {
                const result = JSON.parse(event.data);
                this.updateUI(result);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.handleError(error);
            };
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.handlePermissionDenied();
        }
    }
    
    stop() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        if (this.ws) {
            this.ws.close();
        }
    }
    
    updateUI(result) {
        // Update confidence meter, verdict display
        document.getElementById('confidence').innerText = 
            `${(result.confidence * 100).toFixed(1)}%`;
        document.getElementById('verdict').innerText = result.verdict;
        
        // Update color based on verdict
        const meter = document.getElementById('confidence-meter');
        meter.style.backgroundColor = 
            result.verdict === 'AUTHENTIC' ? 'green' : 'red';
    }
}

// Usage
const recorder = new AudioRecorder('ws://localhost:8000/v1/stream/audio');
document.getElementById('start-btn').onclick = () => recorder.start();
document.getElementById('stop-btn').onclick = () => recorder.stop();
```

**HTML UI:**
```html
<!-- frontend/audio_recording.html -->

<div class="recorder-container">
    <h2>Live Audio Analysis</h2>
    
    <div class="status">
        <span id="recording-indicator" class="indicator"></span>
        <span id="status-text">Ready</span>
    </div>
    
    <div class="confidence-display">
        <div class="label">Authenticity Confidence</div>
        <div id="confidence-meter" class="meter">
            <div id="confidence-bar" class="bar"></div>
        </div>
        <div id="confidence" class="value">--</div>
    </div>
    
    <div class="verdict-display">
        <div class="label">Current Verdict</div>
        <div id="verdict" class="verdict">--</div>
    </div>
    
    <div class="controls">
        <button id="start-btn" class="btn btn-primary">Start Recording</button>
        <button id="stop-btn" class="btn btn-secondary" disabled>Stop</button>
        <button id="pause-btn" class="btn btn-secondary" disabled>Pause</button>
    </div>
    
    <div class="info">
        <p>⚠️ This will access your microphone</p>
        <p>Analysis happens in real-time</p>
    </div>
</div>
```

### Phase 3: Streaming Audio Analysis (Week 2, Days 1-3)

**Tasks:**
- [ ] Implement sliding window buffer
- [ ] Add real-time feature extraction
- [ ] Create streaming inference pipeline
- [ ] Implement verdict smoothing/aggregation

**Backend Implementation:**

```python
# services/api/app/streaming_detector.py (NEW)

import numpy as np
from collections import deque
import torchaudio

class StreamingAudioDetector:
    def __init__(self, model_path: str, window_size: int = 5):
        self.model = load_onnx_model(model_path)
        self.window_size = window_size  # seconds
        self.sample_rate = 16000
        
        # Sliding window buffer
        self.buffer = deque(maxlen=self.window_size * self.sample_rate)
        
        # Smoothing for verdicts
        self.recent_predictions = deque(maxlen=10)
    
    async def process_chunk(self, audio_bytes: bytes) -> dict:
        """Process incoming audio chunk and return verdict."""
        
        # Convert bytes to numpy array
        audio_array = self.bytes_to_audio(audio_bytes)
        
        # Add to sliding window
        self.buffer.extend(audio_array)
        
        # Only analyze if we have enough data
        if len(self.buffer) < self.window_size * self.sample_rate:
            return {
                "status": "buffering",
                "buffer_progress": len(self.buffer) / (self.window_size * self.sample_rate)
            }
        
        # Extract features from buffer
        features = self.extract_features(np.array(self.buffer))
        
        # Run inference
        logits = self.model.run(None, {"audio_features": features})[0]
        confidence = self.softmax(logits)[0][0]  # Probability of "authentic"
        
        # Smooth predictions
        self.recent_predictions.append(confidence)
        smoothed_confidence = np.mean(self.recent_predictions)
        
        # Determine verdict
        if smoothed_confidence > 0.7:
            verdict = "AUTHENTIC"
        elif smoothed_confidence > 0.3:
            verdict = "SUSPICIOUS"
        else:
            verdict = "DEEPFAKE"
        
        return {
            "status": "analyzing",
            "verdict": verdict,
            "confidence": float(smoothed_confidence),
            "buffer_size": len(self.buffer),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def extract_features(self, audio: np.ndarray):
        """Extract log-mel spectrogram features."""
        # Same as training pipeline
        audio_tensor = torch.from_numpy(audio).float()
        features = self.feature_extractor(audio_tensor)
        return features.unsqueeze(0).numpy()
```

**Testing:**
```python
# tests/test_streaming_detector.py

def test_streaming_detector_buffering():
    detector = StreamingAudioDetector("model.onnx", window_size=5)
    
    # Send chunks until buffer is full
    for i in range(10):
        chunk = generate_audio_chunk(1.0)  # 1 second
        result = await detector.process_chunk(chunk)
        
        if i < 5:
            assert result["status"] == "buffering"
        else:
            assert result["status"] == "analyzing"
            assert "verdict" in result

def test_streaming_detector_smoothing():
    # Test that predictions are smoothed over time
    pass

def test_streaming_detector_memory():
    # Test that buffer doesn't grow indefinitely
    pass
```

### Phase 4: Video Recording (Week 2, Days 4-5 & Week 3, Days 1-2)

**Tasks:**
- [ ] Extend MediaRecorder for video
- [ ] Implement frame extraction from video stream
- [ ] Add video streaming endpoint
- [ ] Handle synchronized audio-video streams

**Frontend Code:**

```javascript
// frontend/src/components/VideoRecorder.js

class VideoRecorder {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.mediaRecorder = null;
        this.stream = null;
        this.videoPreview = null;
    }
    
    async start(videoElement) {
        try {
            // Request camera + microphone
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                },
                audio: true
            });
            
            // Show preview
            this.videoPreview = videoElement;
            this.videoPreview.srcObject = this.stream;
            await this.videoPreview.play();
            
            // Setup MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'video/webm;codecs=vp9,opus',
                videoBitsPerSecond: 2500000  // 2.5 Mbps
            });
            
            // Setup WebSocket
            this.ws = new WebSocket(this.wsUrl);
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(event.data);
                }
            };
            
            this.ws.onmessage = (event) => {
                const result = JSON.parse(event.data);
                this.updateUI(result);
            };
            
            this.ws.onopen = () => {
                this.mediaRecorder.start(2000); // Chunk every 2s
            };
            
        } catch (error) {
            console.error('Failed to start video recording:', error);
            this.handleError(error);
        }
    }
    
    stop() {
        if (this.mediaRecorder) {
            this.mediaRecorder.stop();
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        if (this.videoPreview) {
            this.videoPreview.srcObject = null;
        }
        
        if (this.ws) {
            this.ws.close();
        }
    }
}
```

### Phase 5: Optimization & Polish (Week 3, Days 3-5)

**Tasks:**
- [ ] Add connection recovery (reconnect on disconnect)
- [ ] Implement bandwidth adaptation
- [ ] Add compression for network efficiency
- [ ] Create recording session history
- [ ] Add export functionality (save recording)

**Testing:**
```python
# tests/test_streaming_e2e.py

async def test_full_audio_streaming_session():
    """End-to-end test of audio streaming."""
    async with AsyncClient(app=app) as client:
        async with client.websocket_connect("/v1/stream/audio") as ws:
            # Simulate 10 seconds of streaming
            for i in range(10):
                chunk = generate_test_audio_chunk(1.0)
                await ws.send_bytes(chunk)
                
                response = await ws.receive_json()
                
                if i < 5:
                    assert response["status"] == "buffering"
                else:
                    assert response["status"] == "analyzing"
                    assert response["verdict"] in ["AUTHENTIC", "SUSPICIOUS", "DEEPFAKE"]
            
            # Close gracefully
            await ws.close()

async def test_concurrent_streaming_sessions():
    """Test multiple users streaming simultaneously."""
    tasks = []
    for i in range(10):
        tasks.append(simulate_streaming_session(duration=30))
    
    results = await asyncio.gather(*tasks)
    assert all(r["success"] for r in results)
```

---

## 4. Testing Strategy

### Unit Tests (40+ tests)
- WebSocket connection/disconnection
- Buffer management (sliding window)
- Feature extraction from chunks
- Verdict smoothing algorithm
- Permission handling

### Integration Tests (20+ tests)
- End-to-end streaming session
- Audio quality preservation
- Latency measurements
- Error recovery
- Session cleanup

### Performance Tests (10+ tests)
- Concurrent streams (10+ users)
- Memory usage over time
- Network bandwidth consumption
- Latency (chunk → verdict)
- CPU/GPU utilization

### Browser Compatibility Tests
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

---

## 5. Acceptance Criteria

### Definition of Done
- [ ] Audio recording working in all major browsers
- [ ] Video recording working (desktop browsers)
- [ ] Real-time feedback <2s latency
- [ ] Supports 10 concurrent sessions
- [ ] Graceful error handling (permissions, disconnects)
- [ ] All tests passing (90+ tests total)
- [ ] Documentation (user guide, API docs)

### Quality Gates
- [ ] Latency: 95th percentile <2s for audio, <5s for video
- [ ] Memory: <500MB per streaming session
- [ ] Bandwidth: <3 Mbps per video session
- [ ] Browser compatibility: Chrome, Firefox, Safari

---

## 6. Security Considerations

### Data Privacy
- [ ] No recording storage without user consent
- [ ] Clear data retention policy
- [ ] Option to delete recording immediately
- [ ] Encrypted WebSocket connections (WSS)

### Access Control
- [ ] Require authentication for streaming
- [ ] Rate limiting (prevent abuse)
- [ ] Session timeouts (max 30 minutes)

---

## 7. Next Steps After Completion

1. **Mobile Apps:** Native iOS/Android apps for better performance
2. **Edge Processing:** Client-side model for instant feedback
3. **Multi-modal:** Combine audio + video analysis in real-time
4. **Alerts:** Real-time notifications for suspicious content

---

## 8. Resources

**WebRTC Tutorials:**
- https://webrtc.org/getting-started/overview
- https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API

**FastAPI WebSockets:**
- https://fastapi.tiangolo.com/advanced/websockets/

**Libraries:**
- JavaScript: MediaRecorder API, WebSocket API
- Python: FastAPI WebSocket, asyncio, torchaudio

---

**Estimated Effort:** 120-160 hours (1-2 engineers, 2-3 weeks)  
**Team:** 1 Frontend Engineer, 1 Backend Engineer
