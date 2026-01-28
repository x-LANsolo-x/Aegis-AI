"""
Video preprocessing utilities for Aegis-AI.

This module handles:
- Video validation (format, codec, duration)
- Frame extraction
- Face detection
- Video metadata extraction
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from datetime import timedelta

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VideoMetadata:
    """Video file metadata."""
    duration_sec: float
    fps: float
    width: int
    height: int
    total_frames: int
    codec: str
    file_size_bytes: int


@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None


@dataclass
class VideoFrame:
    """Extracted video frame."""
    frame_number: int
    timestamp_sec: float
    image: np.ndarray
    faces: List[FaceDetection]


# ============================================================================
# Video Validation
# ============================================================================

class VideoValidator:
    """Validate video files before processing."""
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    MAX_DURATION_SEC = 600  # 10 minutes
    MAX_FILE_SIZE_MB = 500
    
    @classmethod
    def validate(cls, video_path: Path) -> Dict[str, any]:
        """
        Validate video file.
        
        Returns:
            dict with 'valid' (bool) and 'errors' (list) or 'metadata' (VideoMetadata)
        """
        errors = []
        
        # Check file exists
        if not video_path.exists():
            return {'valid': False, 'errors': ['File not found']}
        
        # Check format
        if video_path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            errors.append(f'Unsupported format: {video_path.suffix}. Supported: {cls.SUPPORTED_FORMATS}')
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > cls.MAX_FILE_SIZE_MB:
            errors.append(f'File too large: {file_size_mb:.1f}MB (max: {cls.MAX_FILE_SIZE_MB}MB)')
        
        # Try to open with OpenCV
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                errors.append('Cannot open video file')
                return {'valid': False, 'errors': errors}
            
            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # Convert codec to string
            codec = ''.join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
            
            duration_sec = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            # Check duration
            if duration_sec > cls.MAX_DURATION_SEC:
                errors.append(f'Video too long: {duration_sec:.1f}s (max: {cls.MAX_DURATION_SEC}s)')
            
            if duration_sec == 0:
                errors.append('Invalid video: duration is 0')
            
            if errors:
                return {'valid': False, 'errors': errors}
            
            metadata = VideoMetadata(
                duration_sec=duration_sec,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                codec=codec,
                file_size_bytes=video_path.stat().st_size
            )
            
            return {'valid': True, 'metadata': metadata}
            
        except Exception as e:
            logger.error(f"Error validating video: {e}")
            errors.append(f'Validation error: {str(e)}')
            return {'valid': False, 'errors': errors}


# ============================================================================
# Frame Extraction
# ============================================================================

class FrameExtractor:
    """Extract frames from video for analysis."""
    
    def __init__(self, sample_rate: float = 1.0):
        """
        Args:
            sample_rate: Frames per second to extract (default: 1 FPS)
        """
        self.sample_rate = sample_rate
    
    def extract_frames(
        self,
        video_path: Path,
        max_frames: Optional[int] = None
    ) -> List[VideoFrame]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (optional)
        
        Returns:
            List of VideoFrame objects
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            frame_interval = int(fps / self.sample_rate) if self.sample_rate < fps else 1
            
            frame_count = 0
            extracted_count = 0
            
            logger.info(f"Extracting frames from {video_path.name} (FPS={fps}, sample_rate={self.sample_rate}, interval={frame_interval})")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at interval
                if frame_count % frame_interval == 0:
                    timestamp_sec = frame_count / fps
                    
                    video_frame = VideoFrame(
                        frame_number=frame_count,
                        timestamp_sec=timestamp_sec,
                        image=frame,
                        faces=[]  # Will be populated by face detector
                    )
                    
                    frames.append(video_frame)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        logger.info(f"Reached max_frames limit: {max_frames}")
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from {total_frames} total frames")
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return frames


# ============================================================================
# Face Detection
# ============================================================================

class FaceDetector:
    """Detect faces in video frames using OpenCV Haar Cascades."""
    
    def __init__(self):
        """Initialize face detector with Haar Cascade."""
        # Use OpenCV's pre-trained Haar Cascade (lightweight)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.warning("Failed to load Haar Cascade for face detection")
    
    def detect_faces(self, frame: np.ndarray, min_size=(30, 30)) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Image frame (BGR format)
            min_size: Minimum face size (width, height)
        
        Returns:
            List of FaceDetection objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detection = FaceDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=1.0  # Haar cascade doesn't provide confidence
            )
            detections.append(detection)
        
        return detections
    
    def detect_faces_in_frames(self, frames: List[VideoFrame]) -> List[VideoFrame]:
        """
        Detect faces in all frames.
        
        Args:
            frames: List of VideoFrame objects
        
        Returns:
            Same list with faces populated
        """
        logger.info(f"Detecting faces in {len(frames)} frames")
        
        for frame in frames:
            frame.faces = self.detect_faces(frame.image)
        
        total_faces = sum(len(f.faces) for f in frames)
        logger.info(f"Detected {total_faces} faces across {len(frames)} frames")
        
        return frames
    
    def crop_face(self, frame: np.ndarray, face: FaceDetection, padding: float = 0.2) -> np.ndarray:
        """
        Crop face region with padding.
        
        Args:
            frame: Full frame image
            face: FaceDetection object
            padding: Padding ratio around face bbox
        
        Returns:
            Cropped face image
        """
        x, y, w, h = face.bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        face_crop = frame[y1:y2, x1:x2]
        
        return face_crop


# ============================================================================
# Video Processing Pipeline
# ============================================================================

class VideoProcessor:
    """Complete video processing pipeline."""
    
    def __init__(self, sample_rate: float = 1.0):
        """
        Args:
            sample_rate: Frames per second to extract
        """
        self.frame_extractor = FrameExtractor(sample_rate=sample_rate)
        self.face_detector = FaceDetector()
    
    def process_video(
        self,
        video_path: Path,
        max_frames: Optional[int] = None
    ) -> Tuple[VideoMetadata, List[VideoFrame]]:
        """
        Complete video processing pipeline.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract
        
        Returns:
            (VideoMetadata, List of VideoFrame with detected faces)
        """
        # Validate video
        validation_result = VideoValidator.validate(video_path)
        if not validation_result['valid']:
            raise ValueError(f"Invalid video: {validation_result['errors']}")
        
        metadata = validation_result['metadata']
        
        # Extract frames
        frames = self.frame_extractor.extract_frames(video_path, max_frames)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Detect faces
        frames = self.face_detector.detect_faces_in_frames(frames)
        
        return metadata, frames


# ============================================================================
# Utility Functions
# ============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def get_video_info(video_path: Path) -> Dict:
    """Get video information for display."""
    validation = VideoValidator.validate(video_path)
    
    if not validation['valid']:
        return {
            'valid': False,
            'errors': validation['errors']
        }
    
    metadata = validation['metadata']
    
    return {
        'valid': True,
        'duration': format_duration(metadata.duration_sec),
        'resolution': f"{metadata.width}x{metadata.height}",
        'fps': f"{metadata.fps:.2f}",
        'codec': metadata.codec,
        'total_frames': metadata.total_frames,
        'file_size_mb': f"{metadata.file_size_bytes / (1024 * 1024):.2f}"
    }
