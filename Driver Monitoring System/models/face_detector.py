"""
Face Detector Module

This module implements a face detection system using the MTCNN (Multi-task Cascaded
Convolutional Networks) model for reliable face and facial landmark detection.
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

class FaceDetector:
    """
    Face detector class using MTCNN for detecting faces and facial landmarks.
    """
    def __init__(self, min_face_size=40, thresholds=None, device=None):
        """
        Initialize the face detector.
        
        Args:
            min_face_size (int): Minimum face size to detect
            thresholds (list): MTCNN detection thresholds for the three stages
            device (torch.device, optional): Device to run the model on
        """
        # Set default thresholds if not provided
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]  # Default MTCNN thresholds
        
        # ALWAYS use CPU for MTCNN to avoid MPS adaptive pooling error
        self.device = torch.device("cpu")
        print("Using CPU for face detection to avoid MPS adaptive pooling issues")
        
        # Initialize MTCNN
        self.detector = MTCNN(
            image_size=160,  # Standard size for face recognition
            margin=0,  # No margin
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=0.709,  # Scale factor for the image pyramid
            post_process=True,
            device=self.device,
            keep_all=True  # Keep all detected faces
        )
        
        # Performance optimization: Use a detection cascade
        self.last_faces = None
        self.frame_skip_count = 0
    
    def detect_faces(self, frame, optimize_performance=True, frame_skip=2):
        """
        Detect faces in an image.
        
        Args:
            frame (numpy.ndarray): Image in BGR format
            optimize_performance (bool): Whether to use performance optimization
            frame_skip (int): Number of frames to skip detection (for optimization)
            
        Returns:
            list: List of detected faces with bounding boxes and landmarks
        """
        # Performance optimization: Skip detection on some frames
        if optimize_performance and self.last_faces is not None:
            if self.frame_skip_count < frame_skip:
                self.frame_skip_count += 1
                return self.last_faces
            else:
                self.frame_skip_count = 0
        
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        try:
            # Get bounding boxes and landmarks
            boxes, probs, landmarks = self.detector.detect(rgb_frame, landmarks=True)
            
            # Process results
            faces = []
            
            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    # Skip low confidence detections
                    if prob < 0.7:
                        continue
                    
                    # Convert box to [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Extract landmarks
                    landmark_dict = {
                        'left_eye': (landmark[0][0], landmark[0][1]),
                        'right_eye': (landmark[1][0], landmark[1][1]),
                        'nose': (landmark[2][0], landmark[2][1]),
                        'mouth_left': (landmark[3][0], landmark[3][1]),
                        'mouth_right': (landmark[4][0], landmark[4][1])
                    }
                    
                    # Store face information
                    faces.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(prob),
                        'keypoints': landmark_dict
                    })
            
            # Update last faces for optimization
            self.last_faces = faces
            
            return faces
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def extract_face_regions(self, frame, face, padding=0.2):
        """
        Extract face regions (full face, eyes, mouth) from the frame.
        
        Args:
            frame (numpy.ndarray): Image in BGR format
            face (dict): Face information with bounding box and landmarks
            padding (float): Padding percentage for the extracted regions
            
        Returns:
            dict: Dictionary containing extracted regions
        """
        # Extract face information
        x1, y1, x2, y2 = face['box']
        
        # Add padding to face box
        width, height = x2 - x1, y2 - y1
        x1 = max(0, x1 - int(padding * width))
        y1 = max(0, y1 - int(padding * height))
        x2 = min(frame.shape[1], x2 + int(padding * width))
        y2 = min(frame.shape[0], y2 + int(padding * height))
        
        # Extract face region
        face_region = frame[y1:y2, x1:x2]
        
        # Extract eye regions
        left_eye = face['keypoints']['left_eye']
        right_eye = face['keypoints']['right_eye']
        
        # Calculate eye bounding boxes with padding
        eye_width = int(width * 0.15)
        eye_height = int(height * 0.1)
        
        left_eye_x1 = max(0, int(left_eye[0] - eye_width))
        left_eye_y1 = max(0, int(left_eye[1] - eye_height))
        left_eye_x2 = min(frame.shape[1], int(left_eye[0] + eye_width))
        left_eye_y2 = min(frame.shape[0], int(left_eye[1] + eye_height))
        
        right_eye_x1 = max(0, int(right_eye[0] - eye_width))
        right_eye_y1 = max(0, int(right_eye[1] - eye_height))
        right_eye_x2 = min(frame.shape[1], int(right_eye[0] + eye_width))
        right_eye_y2 = min(frame.shape[0], int(right_eye[1] + eye_height))
        
        # Extract eye regions
        left_eye_region = frame[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2]
        right_eye_region = frame[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2]
        
        # Extract mouth region
        mouth_left = face['keypoints']['mouth_left']
        mouth_right = face['keypoints']['mouth_right']
        
        mouth_width = int((mouth_right[0] - mouth_left[0]) * 1.5)
        mouth_height = int(height * 0.15)
        mouth_center_x = (mouth_left[0] + mouth_right[0]) // 2
        mouth_center_y = (mouth_left[1] + mouth_right[1]) // 2
        
        mouth_x1 = max(0, int(mouth_center_x - mouth_width // 2))
        mouth_y1 = max(0, int(mouth_center_y - mouth_height // 2))
        mouth_x2 = min(frame.shape[1], int(mouth_center_x + mouth_width // 2))
        mouth_y2 = min(frame.shape[0], int(mouth_center_y + mouth_height // 2))
        
        mouth_region = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        
        # Return regions
        return {
            'face': face_region,
            'left_eye': left_eye_region,
            'right_eye': right_eye_region,
            'mouth': mouth_region
        }
    
    def draw_detections(self, frame, faces):
        """
        Draw face detections on the frame.
        
        Args:
            frame (numpy.ndarray): Image in BGR format
            faces (list): List of detected faces
            
        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        # Create a copy of the frame
        result = frame.copy()
        
        for face in faces:
            # Extract face information
            x1, y1, x2, y2 = face['box']
            confidence = face['confidence']
            keypoints = face['keypoints']
            
            # Draw face bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(result, f"{confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw facial landmarks
            for point in keypoints.values():
                cv2.circle(result, (int(point[0]), int(point[1])), 2, (0, 0, 255), 2)
        
        return result