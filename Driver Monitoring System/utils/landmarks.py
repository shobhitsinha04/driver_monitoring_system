"""
Landmarks Module

This module provides utilities for facial landmark detection and analysis.
It extends the capabilities of the MTCNN detector with additional landmark
analysis tools for fatigue detection.
"""

import cv2
import numpy as np
from scipy.spatial import distance
from itertools import combinations

class LandmarkAnalyzer:
    """
    Class for analyzing facial landmarks for signs of fatigue.
    """
    def __init__(self):
        """
        Initialize the landmark analyzer.
        """
        # Constants for facial measurements
        self.EYE_AR_THRESHOLD = 0.25  # Eye aspect ratio threshold
        self.MOUTH_AR_THRESHOLD = 0.60  # Mouth aspect ratio threshold
        self.HEAD_POSE_THRESHOLD = 25.0  # Head pose angle threshold (degrees)
        
        # State tracking
        self.prev_landmarks = None
        self.landmark_stability = []
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate the eye aspect ratio (EAR) from landmarks.
        
        Args:
            eye_landmarks (list): List of 6 points representing eye landmarks
            
        Returns:
            float: Eye aspect ratio
        """
        # Calculate distances between vertical landmarks
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Calculate distance between horizontal landmarks
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        """
        Calculate the mouth aspect ratio (MAR) from landmarks.
        
        Args:
            mouth_landmarks (list): List of points representing mouth landmarks
            
        Returns:
            float: Mouth aspect ratio
        """
        # Calculate vertical distances
        A = distance.euclidean(mouth_landmarks[2], mouth_landmarks[10])
        B = distance.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        C = distance.euclidean(mouth_landmarks[0], mouth_landmarks[6])
        
        # Calculate MAR
        mar = (A + B) / (2.0 * C)
        
        return mar
    
    def estimate_head_pose(self, face_landmarks, image_size):
        """
        Estimate head pose angle from landmarks.
        
        Args:
            face_landmarks (dict): Dictionary of facial landmarks
            image_size (tuple): Size of the image (width, height)
            
        Returns:
            tuple: (roll, pitch, yaw) angles in degrees
        """
        # Get key landmarks
        left_eye = np.array(face_landmarks['left_eye'])
        right_eye = np.array(face_landmarks['right_eye'])
        nose = np.array(face_landmarks['nose'])
        mouth_left = np.array(face_landmarks['mouth_left'])
        mouth_right = np.array(face_landmarks['mouth_right'])
        
        # Calculate center of the face
        face_center = np.mean([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)
        
        # Calculate eye line angle (roll)
        eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        roll = np.degrees(eye_angle)
        
        # Estimate pitch from vertical position of nose relative to eyes
        eye_center = (left_eye + right_eye) / 2
        vertical_ratio = (nose[1] - eye_center[1]) / (image_size[1] * 0.3)
        pitch = (vertical_ratio - 0.5) * 90
        
        # Estimate yaw from horizontal position of nose relative to eyes
        horizontal_ratio = (nose[0] - face_center[0]) / (image_size[0] * 0.15)
        yaw = horizontal_ratio * 90
        
        return roll, pitch, yaw
    
    def analyze_landmark_stability(self, landmarks):
        """
        Analyze stability of landmarks across frames.
        
        Args:
            landmarks (dict): Dictionary of facial landmarks
            
        Returns:
            float: Stability score (0-1, higher is more stable)
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return 1.0
        
        # Calculate movement for each landmark
        movements = []
        
        for key in landmarks:
            if key in self.prev_landmarks:
                curr = np.array(landmarks[key])
                prev = np.array(self.prev_landmarks[key])
                movement = np.linalg.norm(curr - prev)
                movements.append(movement)
        
        # Calculate average movement
        avg_movement = np.mean(movements) if movements else 0
        
        # Convert to stability score (inverse of movement)
        stability = max(0, 1.0 - (avg_movement / 10.0))
        
        # Update stability history
        self.landmark_stability.append(stability)
        if len(self.landmark_stability) > 10:
            self.landmark_stability.pop(0)
        
        # Update previous landmarks
        self.prev_landmarks = landmarks
        
        # Return average stability
        return np.mean(self.landmark_stability)
    
    def detect_drowsiness_from_landmarks(self, landmarks, image_size):
        """
        Detect drowsiness from facial landmarks.
        
        Args:
            landmarks (dict): Dictionary of facial landmarks
            image_size (tuple): Size of the image (width, height)
            
        Returns:
            dict: Drowsiness indicators and confidence
        """
        # Extract eye landmarks
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        # Extract mouth landmarks
        mouth_left = landmarks['mouth_left']
        mouth_right = landmarks['mouth_right']
        nose = landmarks['nose']
        
        # Create mouth landmarks array
        # This is an approximation since MTCNN doesn't provide all mouth landmarks
        center_y = (mouth_left[1] + mouth_right[1]) / 2
        top_lip_y = center_y - abs(mouth_right[0] - mouth_left[0]) * 0.1
        bottom_lip_y = center_y + abs(mouth_right[0] - mouth_left[0]) * 0.1
        
        mouth_landmarks = [
            mouth_left,
            [mouth_left[0] + (mouth_right[0] - mouth_left[0]) * 0.25, top_lip_y],
            [mouth_left[0] + (mouth_right[0] - mouth_left[0]) * 0.5, top_lip_y],
            [mouth_left[0] + (mouth_right[0] - mouth_left[0]) * 0.75, top_lip_y],
            mouth_right,
            [mouth_left[0] + (mouth_right[0] - mouth_left[0]) * 0.75, bottom_lip_y],
            [mouth_left[0] + (mouth_right[0] - mouth_left[0]) * 0.5, bottom_lip_y],
            [mouth_left[0] + (mouth_right[0] - mouth_left[0]) * 0.25, bottom_lip_y]
        ]
        
        # Construct eye landmarks (approximation)
        left_eye_landmarks = [
            [left_eye[0] - 10, left_eye[1]],
            [left_eye[0] - 5, left_eye[1] - 5],
            [left_eye[0], left_eye[1] - 5],
            [left_eye[0] + 10, left_eye[1]],
            [left_eye[0], left_eye[1] + 5],
            [left_eye[0] - 5, left_eye[1] + 5]
        ]
        
        right_eye_landmarks = [
            [right_eye[0] - 10, right_eye[1]],
            [right_eye[0] - 5, right_eye[1] - 5],
            [right_eye[0], right_eye[1] - 5],
            [right_eye[0] + 10, right_eye[1]],
            [right_eye[0], right_eye[1] + 5],
            [right_eye[0] - 5, right_eye[1] + 5]
        ]
        
        # Calculate metrics
        left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mouth_aspect_ratio(mouth_landmarks)
        
        roll, pitch, yaw = self.estimate_head_pose(landmarks, image_size)
        
        stability = self.analyze_landmark_stability(landmarks)
        
        # Evaluate drowsiness indicators
        eyes_closed = avg_ear < self.EYE_AR_THRESHOLD
        yawning = mar > self.MOUTH_AR_THRESHOLD
        head_drooping = abs(pitch) > self.HEAD_POSE_THRESHOLD
        low_stability = stability < 0.7  # Movement can indicate drowsiness
        
        # Calculate overall drowsiness confidence
        confidence = 0.0
        if eyes_closed:
            confidence += 0.5
        if yawning:
            confidence += 0.3
        if head_drooping:
            confidence += 0.3
        if low_stability:
            confidence += 0.1
        
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)
        
        return {
            "eyes_closed": eyes_closed,
            "yawning": yawning,
            "head_drooping": head_drooping,
            "low_stability": low_stability,
            "ear": avg_ear,
            "mar": mar,
            "head_pose": (roll, pitch, yaw),
            "stability": stability,
            "confidence": confidence
        }
    
    def draw_landmarks(self, frame, landmarks, drowsiness_indicators=None):
        """
        Draw landmarks and drowsiness indicators on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            landmarks (dict): Dictionary of facial landmarks
            drowsiness_indicators (dict, optional): Drowsiness indicators
            
        Returns:
            numpy.ndarray: Frame with landmarks drawn
        """
        # Create a copy of the frame
        result = frame.copy()
        
        # Draw landmarks
        for key, point in landmarks.items():
            x, y = int(point[0]), int(point[1])
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1)
        
        # Draw drowsiness indicators if available
        if drowsiness_indicators is not None:
            # Determine color based on drowsiness confidence
            confidence = drowsiness_indicators["confidence"]
            color = (0, int(255 * (1 - confidence)), int(255 * confidence))
            
            # Draw indicators
            if drowsiness_indicators["eyes_closed"]:
                cv2.putText(result, "Eyes: Closed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(result, "Eyes: Open", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if drowsiness_indicators["yawning"]:
                cv2.putText(result, "Yawning: Yes", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if drowsiness_indicators["head_drooping"]:
                cv2.putText(result, "Head Drooping", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw confidence
            cv2.putText(result, f"Drowsiness: {confidence:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result