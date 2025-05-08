"""
Models package for the Driver Fatigue Detection System.

This package contains modules for face detection and fatigue detection.
"""

from .face_detector import FaceDetector
from .fatigue_detector import FatigueDetector, EyeStateClassifier, EyeDataset

__all__ = ['FaceDetector', 'FatigueDetector', 'EyeStateClassifier', 'EyeDataset']