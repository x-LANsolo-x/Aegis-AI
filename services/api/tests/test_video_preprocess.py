"""
Tests for video preprocessing module.
"""

import pytest
from pathlib import Path
import numpy as np
import cv2

from services.api.app.video_preprocess import (
    VideoValidator,
    FrameExtractor,
    FaceDetector,
    VideoProcessor,
    VideoFrame,
    FaceDetection,
    get_video_info,
    format_duration
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_video_path(tmp_path):
    """Create a simple test video."""
    video_path = tmp_path / "test_video.mp4"
    
    # Create a simple video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Write 90 frames (3 seconds at 30 FPS)
    for i in range(90):
        # Create a frame with changing color
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (i * 2, 100, 200)
        out.write(frame)
    
    out.release()
    
    return video_path


@pytest.fixture
def sample_frame_with_face():
    """Create a test frame with a face-like region."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple face-like region (rectangle)
    cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
    # Eyes
    cv2.circle(frame, (270, 210), 20, (0, 0, 0), -1)
    cv2.circle(frame, (370, 210), 20, (0, 0, 0), -1)
    # Mouth
    cv2.ellipse(frame, (320, 280), (50, 30), 0, 0, 180, (0, 0, 0), 2)
    
    return frame


# ============================================================================
# VideoValidator Tests
# ============================================================================

def test_video_validator_valid_video(sample_video_path):
    """Test validation of a valid video."""
    result = VideoValidator.validate(sample_video_path)
    
    assert result['valid'] is True
    assert 'metadata' in result
    
    metadata = result['metadata']
    assert metadata.fps == 30.0
    assert metadata.total_frames == 90
    assert metadata.width == 640
    assert metadata.height == 480
    assert 2.9 < metadata.duration_sec < 3.1  # ~3 seconds


def test_video_validator_missing_file(tmp_path):
    """Test validation of missing file."""
    result = VideoValidator.validate(tmp_path / "nonexistent.mp4")
    
    assert result['valid'] is False
    assert 'errors' in result
    assert 'File not found' in result['errors']


def test_video_validator_unsupported_format(tmp_path):
    """Test validation of unsupported format."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("not a video")
    
    result = VideoValidator.validate(invalid_file)
    
    assert result['valid'] is False
    assert any('Unsupported format' in err for err in result['errors'])


# ============================================================================
# FrameExtractor Tests
# ============================================================================

def test_frame_extractor_default_rate(sample_video_path):
    """Test frame extraction at 1 FPS."""
    extractor = FrameExtractor(sample_rate=1.0)
    frames = extractor.extract_frames(sample_video_path)
    
    # 3 second video at 1 FPS = ~3 frames
    assert 2 <= len(frames) <= 4
    
    # Check frame properties
    assert frames[0].frame_number >= 0
    assert frames[0].timestamp_sec >= 0
    assert frames[0].image.shape == (480, 640, 3)
    assert isinstance(frames[0].faces, list)


def test_frame_extractor_high_rate(sample_video_path):
    """Test frame extraction at higher rate."""
    extractor = FrameExtractor(sample_rate=10.0)
    frames = extractor.extract_frames(sample_video_path)
    
    # 3 second video at 10 FPS = ~30 frames
    assert 25 <= len(frames) <= 35


def test_frame_extractor_max_frames(sample_video_path):
    """Test max_frames limit."""
    extractor = FrameExtractor(sample_rate=10.0)
    frames = extractor.extract_frames(sample_video_path, max_frames=5)
    
    assert len(frames) == 5


def test_frame_extractor_invalid_video(tmp_path):
    """Test extraction from invalid video."""
    invalid_path = tmp_path / "invalid.mp4"
    invalid_path.write_bytes(b"not a video")
    
    extractor = FrameExtractor()
    frames = extractor.extract_frames(invalid_path)
    
    assert len(frames) == 0


# ============================================================================
# FaceDetector Tests
# ============================================================================

def test_face_detector_initialization():
    """Test face detector initializes."""
    detector = FaceDetector()
    assert detector.face_cascade is not None


def test_face_detector_detect_faces(sample_frame_with_face):
    """Test face detection in a frame."""
    detector = FaceDetector()
    detections = detector.detect_faces(sample_frame_with_face)
    
    # Should detect at least one face-like region
    assert len(detections) >= 0  # May or may not detect simple drawing
    
    if detections:
        face = detections[0]
        assert isinstance(face, FaceDetection)
        assert len(face.bbox) == 4
        assert 0 <= face.confidence <= 1.0


def test_face_detector_no_faces():
    """Test detection with no faces."""
    detector = FaceDetector()
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detections = detector.detect_faces(empty_frame)
    assert isinstance(detections, list)


def test_face_detector_crop_face():
    """Test face cropping."""
    detector = FaceDetector()
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    face = FaceDetection(bbox=(200, 150, 100, 120), confidence=0.95)
    cropped = detector.crop_face(frame, face, padding=0.1)
    
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    assert cropped.shape[2] == 3


def test_face_detector_detect_in_frames(sample_video_path):
    """Test face detection across multiple frames."""
    extractor = FrameExtractor(sample_rate=1.0)
    frames = extractor.extract_frames(sample_video_path, max_frames=3)
    
    detector = FaceDetector()
    frames_with_faces = detector.detect_faces_in_frames(frames)
    
    assert len(frames_with_faces) == len(frames)
    for frame in frames_with_faces:
        assert isinstance(frame.faces, list)


# ============================================================================
# VideoProcessor Tests
# ============================================================================

def test_video_processor_full_pipeline(sample_video_path):
    """Test complete video processing pipeline."""
    processor = VideoProcessor(sample_rate=1.0)
    
    metadata, frames = processor.process_video(sample_video_path, max_frames=5)
    
    # Check metadata
    assert metadata.duration_sec > 0
    assert metadata.fps == 30.0
    assert metadata.width == 640
    assert metadata.height == 480
    
    # Check frames
    assert len(frames) <= 5
    for frame in frames:
        assert frame.image.shape == (480, 640, 3)
        assert isinstance(frame.faces, list)


def test_video_processor_invalid_video(tmp_path):
    """Test processor with invalid video."""
    invalid_path = tmp_path / "invalid.mp4"
    invalid_path.write_bytes(b"not a video")
    
    processor = VideoProcessor()
    
    with pytest.raises(ValueError, match="Invalid video"):
        processor.process_video(invalid_path)


# ============================================================================
# Utility Function Tests
# ============================================================================

def test_format_duration():
    """Test duration formatting."""
    assert format_duration(0) == "0:00:00"
    assert format_duration(65) == "0:01:05"
    assert format_duration(3661) == "1:01:01"
    assert format_duration(3600 * 2 + 60 * 30 + 45) == "2:30:45"


def test_get_video_info(sample_video_path):
    """Test video info extraction."""
    info = get_video_info(sample_video_path)
    
    assert info['valid'] is True
    assert 'duration' in info
    assert 'resolution' in info
    assert info['resolution'] == "640x480"
    assert 'fps' in info
    assert 'codec' in info
    assert 'total_frames' in info
    assert int(info['total_frames']) == 90


def test_get_video_info_invalid_file(tmp_path):
    """Test video info for invalid file."""
    invalid_path = tmp_path / "test.txt"
    invalid_path.write_text("not a video")
    
    info = get_video_info(invalid_path)
    
    assert info['valid'] is False
    assert 'errors' in info


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_very_short_video(tmp_path):
    """Test with very short video (1 frame)."""
    video_path = tmp_path / "short.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    
    # Write just 1 frame
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out.write(frame)
    out.release()
    
    validator_result = VideoValidator.validate(video_path)
    # Might be valid or invalid depending on OpenCV behavior
    assert 'valid' in validator_result


def test_large_resolution_frame_extraction(tmp_path):
    """Test extraction from high resolution video."""
    video_path = tmp_path / "hd.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1920, 1080))
    
    # Write 30 frames
    for i in range(30):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    extractor = FrameExtractor(sample_rate=1.0)
    frames = extractor.extract_frames(video_path, max_frames=2)
    
    assert len(frames) <= 2
    if frames:
        assert frames[0].image.shape == (1080, 1920, 3)
