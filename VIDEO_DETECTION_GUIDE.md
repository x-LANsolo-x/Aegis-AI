# Video Deepfake Detection - Implementation Guide

## Feature: A1 - Video Deepfake Detection (Spatial Features)

**Priority:** High  
**Estimated Time:** 3-4 weeks  
**Dependencies:** Audio detection complete, PyTorch environment

---

## 1. Requirements

### Functional Requirements
- [ ] Accept video file uploads (MP4, AVI, MOV formats)
- [ ] Extract frames from video for analysis
- [ ] Detect spatial artifacts (face manipulation, blending errors)
- [ ] Provide per-frame and overall video verdict
- [ ] Return confidence scores and heatmaps
- [ ] Support videos up to 10 minutes in length

### Non-Functional Requirements
- [ ] Process 1080p video in <30 seconds
- [ ] Memory usage <2GB during processing
- [ ] Support concurrent analysis (up to 10 videos)
- [ ] Graceful handling of corrupted/invalid videos

### Technical Requirements
- [ ] OpenCV or FFmpeg for video processing
- [ ] Face detection model (MTCNN or RetinaFace)
- [ ] Deepfake detection model (Xception-based or EfficientNet)
- [ ] GPU acceleration support
- [ ] ONNX export for production deployment

---

## 2. Architecture

### Component Overview

```
┌─────────────────┐
│  Video Upload   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Frame Extractor │ (FFmpeg/OpenCV)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Face Detector  │ (MTCNN/RetinaFace)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Frame Analyzer  │ (CNN Model)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Aggregator    │ (Voting/Average)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Video Verdict   │
└─────────────────┘
```

### Data Flow

1. **Upload**: Video → Validation → Storage
2. **Extraction**: Video → Frames (1 FPS or key frames)
3. **Detection**: Frame → Face bounding boxes
4. **Analysis**: Face crops → CNN → Predictions
5. **Aggregation**: Per-frame scores → Overall verdict
6. **Response**: Verdict + confidence + metadata

### Database Schema Changes

```sql
-- New table for video-specific metadata
CREATE TABLE video_metadata (
    analysis_id UUID PRIMARY KEY REFERENCES analysis(id),
    duration_sec FLOAT NOT NULL,
    fps FLOAT NOT NULL,
    resolution TEXT NOT NULL,  -- e.g., "1920x1080"
    codec TEXT,
    num_frames_analyzed INTEGER NOT NULL,
    faces_detected INTEGER NOT NULL
);

-- New table for per-frame results
CREATE TABLE frame_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analysis(id),
    frame_number INTEGER NOT NULL,
    timestamp_sec FLOAT NOT NULL,
    verdict TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    face_bbox JSONB,  -- {x, y, w, h}
    artifacts_detected JSONB  -- Array of artifact types
);
```

---

## 3. Implementation Phases

### Phase 1: Video Processing Infrastructure (Week 1)

**Tasks:**
- [ ] Add video validation (format, codec, duration checks)
- [ ] Implement frame extraction utility
- [ ] Add face detection pipeline
- [ ] Create frame storage/caching mechanism

**Files to Create/Modify:**
- `services/api/app/video_preprocess.py` (NEW)
- `services/api/app/media.py` (UPDATE - add video support)
- `services/api/app/models.py` (UPDATE - add VideoMetadata)

**Testing:**
```python
# tests/test_video_preprocess.py
def test_extract_frames_from_video():
    frames = extract_frames("test.mp4", fps=1)
    assert len(frames) > 0
    assert frames[0].shape == (1080, 1920, 3)

def test_detect_faces_in_frame():
    faces = detect_faces(frame)
    assert len(faces) >= 0
    for face in faces:
        assert "bbox" in face
        assert "confidence" in face
```

### Phase 2: Model Development (Week 2)

**Tasks:**
- [ ] Research state-of-the-art video deepfake detection
- [ ] Select base architecture (Xception, EfficientNet, or Vision Transformer)
- [ ] Prepare training dataset (FaceForensics++, Celeb-DF, DFDC)
- [ ] Implement data augmentation pipeline
- [ ] Train initial model

**Datasets to Use:**
1. **FaceForensics++** (Primary)
   - 1000 real videos, 4000 fake videos
   - Multiple manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
   - Download: https://github.com/ondyari/FaceForensics

2. **Celeb-DF** (Secondary)
   - High-quality deepfakes
   - 590 real, 5639 fake videos
   - Better represents real-world scenarios

**Model Architecture:**
```python
# ml/training/models/video_detector.py
class SpatialDeepfakeDetector(nn.Module):
    def __init__(self, backbone='xception'):
        super().__init__()
        if backbone == 'xception':
            self.backbone = xception(pretrained=True)
        
        # Replace classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Binary: real/fake
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(pooled)
```

**Training Script:**
```bash
python ml/training/train_video.py \
    --dataset faceforensics \
    --model xception \
    --epochs 50 \
    --batch-size 32 \
    --augmentation heavy
```

**Testing:**
```python
# ml/tests/test_video_model.py
def test_model_forward_pass():
    model = SpatialDeepfakeDetector()
    x = torch.randn(4, 3, 299, 299)  # Batch of 4 face crops
    output = model(x)
    assert output.shape == (4, 2)

def test_model_onnx_export():
    export_to_onnx(model, "video_detector.onnx")
    assert Path("video_detector.onnx").exists()
    verify_onnx_numeric_parity(model, "video_detector.onnx")
```

### Phase 3: API Integration (Week 3)

**Tasks:**
- [ ] Add video upload endpoint
- [ ] Implement video analysis workflow
- [ ] Create per-frame analysis storage
- [ ] Add video verdict aggregation logic
- [ ] Update schemas and responses

**API Changes:**

**New Endpoints:**
```python
POST /v1/analysis/upload
# Updated to accept video files
# Request: multipart/form-data with video file
# Response: analysis_id + video_metadata

GET /v1/analysis/{id}/frames
# New endpoint to retrieve per-frame results
# Response: List of frame analyses

GET /v1/analysis/{id}/heatmap
# New endpoint for visualization
# Response: Video with heatmap overlay
```

**Updated Schemas:**
```python
# services/api/app/schemas.py

class VideoMetadata(BaseModel):
    duration_sec: float
    fps: float
    resolution: str
    codec: str
    num_frames_analyzed: int
    faces_detected: int

class FrameAnalysis(BaseModel):
    frame_number: int
    timestamp_sec: float
    verdict: Verdict
    confidence: float
    face_bbox: Optional[dict]
    artifacts_detected: List[str]

class AnalysisResponse(BaseModel):
    # ... existing fields ...
    video_metadata: Optional[VideoMetadata]
    frame_analyses: Optional[List[FrameAnalysis]]
```

**Testing:**
```python
# tests/test_video_endpoints.py
def test_upload_video(client):
    with open("test_video.mp4", "rb") as f:
        response = client.post("/v1/analysis/upload", files={"file": f})
    assert response.status_code == 200
    assert "video_metadata" in response.json()

def test_analyze_video(client, video_analysis_id):
    response = client.post(f"/v1/analysis/{video_analysis_id}/run")
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ["AUTHENTIC", "SUSPICIOUS", "DEEPFAKE"]
    assert "video_metadata" in data

def test_get_frame_analyses(client, video_analysis_id):
    response = client.get(f"/v1/analysis/{video_analysis_id}/frames")
    assert response.status_code == 200
    frames = response.json()
    assert len(frames) > 0
    assert "frame_number" in frames[0]
```

### Phase 4: Performance Optimization (Week 4)

**Tasks:**
- [ ] Add frame batching for GPU efficiency
- [ ] Implement caching for repeated analyses
- [ ] Optimize face detection (skip frames without faces)
- [ ] Add progress reporting for long videos
- [ ] Load testing and profiling

**Performance Targets:**
- 1080p, 30 FPS, 1-minute video: <30 seconds total
- Memory peak: <2GB
- Concurrent videos: 10 simultaneous

**Testing:**
```python
# tests/test_video_performance.py
def test_video_processing_latency():
    start = time.time()
    result = analyze_video("test_1min.mp4")
    duration = time.time() - start
    assert duration < 30, f"Processing took {duration}s"

def test_memory_usage():
    mem_before = get_memory_usage()
    analyze_video("test_1080p.mp4")
    mem_after = get_memory_usage()
    assert (mem_after - mem_before) < 2000  # MB
```

---

## 4. Testing Strategy

### Unit Tests (50+ tests)
- Frame extraction edge cases
- Face detection accuracy
- Model inference correctness
- Aggregation logic
- Schema validation

### Integration Tests (20+ tests)
- End-to-end video upload → analysis → result
- Database persistence
- Error handling (corrupted videos, no faces)
- Concurrent processing

### Performance Tests (10+ tests)
- Latency benchmarks (various video lengths)
- Memory stability
- Throughput (videos/minute)
- GPU utilization

### Acceptance Tests (5+ tests)
- Real video samples (deepfake + authentic)
- Golden dataset (known verdicts)
- Visual inspection of heatmaps
- User scenario walkthroughs

---

## 5. Acceptance Criteria

### Definition of Done
- [ ] Video upload working for MP4, AVI, MOV
- [ ] Model achieves >85% accuracy on FaceForensics++
- [ ] API responds in <30s for 1-minute 1080p video
- [ ] All tests passing (unit + integration + E2E)
- [ ] Documentation updated (API spec, user guide)
- [ ] Code reviewed and merged to main branch

### Quality Gates
- [ ] Test coverage >80%
- [ ] No memory leaks in 1-hour stress test
- [ ] Performance within SLA (95th percentile <30s)
- [ ] Security review passed (file validation, injection)

---

## 6. Risks & Mitigations

### Risk 1: Model Performance
**Issue:** Video deepfake detection is harder than audio  
**Mitigation:** Start with FaceForensics++ (easier), use pretrained weights, ensemble methods

### Risk 2: Computational Cost
**Issue:** Video processing is CPU/GPU intensive  
**Mitigation:** Frame sampling, batching, GPU acceleration, cloud scaling

### Risk 3: Dataset Availability
**Issue:** FaceForensics++ requires academic request  
**Mitigation:** Start with public subsets, use Celeb-DF, synthetic data initially

### Risk 4: Face Detection Failures
**Issue:** Videos with no faces or poor quality  
**Mitigation:** Fallback to full-frame analysis, clear error messages

---

## 7. Next Steps After Completion

1. **Temporal Analysis (A2):** Add LSTM/Transformer for inter-frame inconsistencies
2. **Audio-Video Sync (A3):** Detect lip-sync mismatches
3. **Explainability:** Generate attention maps showing manipulation regions
4. **Edge Deployment:** Optimize for mobile/edge devices

---

## 8. Resources

**Papers to Review:**
- FaceForensics++: Learning to Detect Manipulated Facial Images (2019)
- Detecting Face Synthesis Using Convolutional Neural Networks (2018)
- The Eyes Tell All: Detecting Political Orientation from Eye Movement Data (2020)

**Code References:**
- https://github.com/ondyari/FaceForensics
- https://github.com/yuezunli/WIFS2018_In_Ictu_Oculi
- https://github.com/HongguLiu/EfficientFace

**Tools:**
- FFmpeg for video processing
- MTCNN for face detection
- Xception/EfficientNet for classification
- Grad-CAM for visualization

---

**Estimated Effort:** 160-200 hours (1 engineer, 4 weeks)  
**Team:** 1 ML Engineer, 1 Backend Engineer (code review)
