"""
Eye Analyzer Module

This module provides utilities for analyzing eye state and detecting blinks
using facial landmarks and image processing techniques.
"""

import cv2
import numpy as np
from scipy.spatial import distance

class EyeAnalyzer:
    """
    Class for analyzing eye state and detecting blinks.
    """
    def __init__(self, eye_aspect_ratio_threshold=0.25, consecutive_frames=3):
        """
        Initialize the eye analyzer.
        
        Args:
            eye_aspect_ratio_threshold (float): Threshold for determining eye closure
            consecutive_frames (int): Number of consecutive frames for blink detection
        """
        self.ear_threshold = eye_aspect_ratio_threshold
        self.consecutive_frames = consecutive_frames
        
        # State variables for blink detection
        self.eye_closed_frames = 0
        self.eye_just_closed = False
        self.eye_just_opened = False
        self.last_ear = 1.0
        
        # Blink statistics
        self.total_blinks = 0
        self.blink_start_time = None
        self.blink_durations = []
    
    def extract_eye_regions(self, frame, face):
        """
        Extract the eye regions from a frame using facial landmarks.
        
        Args:
            frame (numpy.ndarray): Frame containing the face
            face (dict): Face information with landmarks
            
        Returns:
            tuple: Left and right eye regions as numpy arrays
        """
        try:
            # Get face dimensions
            x1, y1, x2, y2 = face['box']
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Get eye landmarks
            left_eye = face['keypoints']['left_eye']
            right_eye = face['keypoints']['right_eye']
            
            # Calculate eye region size (proportional to face size)
            eye_width = int(face_width * 0.18)  # Increase width
            eye_height = int(face_height * 0.12)
            
            # Calculate eye region boundaries
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
            
            # Check if regions are valid
            if (left_eye_region.size == 0 or right_eye_region.size == 0 or
                left_eye_region.shape[0] < 4 or left_eye_region.shape[1] < 4 or
                right_eye_region.shape[0] < 4 or right_eye_region.shape[1] < 4):
                return None, None
            
            return left_eye_region, right_eye_region
            
        except Exception as e:
            print(f"Error extracting eye regions: {e}")
            return None, None
    
    def calculate_eye_aspect_ratio(self, eye_region):
        """
        Calculate the eye aspect ratio (EAR) for an eye region.
        
        EAR = (A + B) / (2 * C)
        
        Where:
        - A is the height of the eye at the center
        - B is the height of the eye at the sides
        - C is the width of the eye
        
        Args:
            eye_region (numpy.ndarray): Eye region image
            
        Returns:
            float: Eye aspect ratio
        """
        if eye_region is None or eye_region.size == 0:
            return 1.0  # Default to open eye
        
        try:
            # Convert to grayscale if it's not already
            if len(eye_region.shape) == 3:
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_eye = eye_region
            
            # Apply thresholding to separate eye from background
            _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 1.0  # Default to open eye
            
            # Find the largest contour (should be the eye)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate EAR
            ear = h / w if w > 0 else 1.0
            
            # Normalize EAR
            ear = min(1.0, ear)
            
            return ear
            
        except Exception as e:
            print(f"Error calculating eye aspect ratio: {e}")
            return 1.0  # Default to open eye
    
    def is_blink(self, ear):
        """
        Detect blinks based on eye aspect ratio.
        
        Args:
            ear (float): Eye aspect ratio
            
        Returns:
            bool: True if a blink is detected, False otherwise
        """
        # Check if eye is closed
        is_closed = ear < self.ear_threshold
        
        # Update state
        if is_closed:
            self.eye_closed_frames += 1
            if self.eye_closed_frames == 1:
                self.eye_just_closed = True
                self.blink_start_time = cv2.getTickCount()
            else:
                self.eye_just_closed = False
            self.eye_just_opened = False
        else:
            if self.eye_closed_frames >= 1 and self.eye_closed_frames <= self.consecutive_frames:
                self.eye_just_opened = True
                
                # Calculate blink duration
                if self.blink_start_time is not None:
                    end_time = cv2.getTickCount()
                    blink_duration = (end_time - self.blink_start_time) / cv2.getTickFrequency() * 1000  # ms
                    self.blink_durations.append(blink_duration)
                
                # Increment blink counter
                self.total_blinks += 1
                
                # Reset blink start time
                self.blink_start_time = None
            else:
                self.eye_just_opened = False
            
            self.eye_closed_frames = 0
            self.eye_just_closed = False
        
        # Store current EAR for next frame
        self.last_ear = ear
        
        # Return True if a blink is detected
        return self.eye_just_opened
    
    def detect_drowsiness(self, ear, period=60, blink_threshold=30):
        """
        Detect drowsiness based on blink frequency and duration.
        
        Args:
            ear (float): Current eye aspect ratio
            period (int): Time period in seconds to consider for blink statistics
            blink_threshold (int): Threshold for number of blinks in the period
            
        Returns:
            tuple: (is_drowsy, confidence)
        """
        # Basic drowsiness detection based on eye closure
        if ear < self.ear_threshold and self.eye_closed_frames > self.consecutive_frames * 3:
            return True, 0.9  # High confidence
        
        # Check blink frequency
        if len(self.blink_durations) >= blink_threshold:
            # Calculate average blink duration
            avg_duration = sum(self.blink_durations[-blink_threshold:]) / blink_threshold
            
            # Drowsiness is indicated by longer blink durations
            if avg_duration > 150:  # ms
                return True, 0.7 + min(0.2, (avg_duration - 150) / 250)  # Scale confidence
        
        return False, 0.0
    
    def enhance_eye_region(self, eye_region):
        """
        Enhance the eye region for better analysis.
        
        Args:
            eye_region (numpy.ndarray): Eye region image
            
        Returns:
            numpy.ndarray: Enhanced eye region
        """
        if eye_region is None or eye_region.size == 0:
            return None
        
        try:
            # Convert to grayscale if it's not already
            if len(eye_region.shape) == 3:
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_eye = eye_region
            
            # Apply histogram equalization to enhance contrast
            enhanced_eye = cv2.equalizeHist(gray_eye)
            
            # Apply Gaussian blur to reduce noise
            enhanced_eye = cv2.GaussianBlur(enhanced_eye, (5, 5), 0)
            
            return enhanced_eye
            
        except Exception as e:
            print(f"Error enhancing eye region: {e}")
            return eye_region
    
    def visualize_eye_state(self, frame, left_eye, right_eye, ear, is_closed):
        """
        Visualize the eye state on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to visualize on
            left_eye (tuple): Left eye coordinates
            right_eye (tuple): Right eye coordinates
            ear (float): Eye aspect ratio
            is_closed (bool): Whether the eye is closed
            
        Returns:
            numpy.ndarray: Frame with visualization
        """
        # Create a copy of the frame
        result = frame.copy()
        
        # Draw circles around eyes
        left_eye_x, left_eye_y = int(left_eye[0]), int(left_eye[1])
        right_eye_x, right_eye_y = int(right_eye[0]), int(right_eye[1])
        
        # Determine color based on eye state
        color = (0, 0, 255) if is_closed else (0, 255, 0)
        
        # Draw circles
        cv2.circle(result, (left_eye_x, left_eye_y), 5, color, 2)
        cv2.circle(result, (right_eye_x, right_eye_y), 5, color, 2)
        
        # Draw EAR value
        cv2.putText(result, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result
    
    def get_blink_statistics(self):
        """
        Get blink statistics.
        
        Returns:
            dict: Blink statistics
        """
        if not self.blink_durations:
            return {
                "total_blinks": 0,
                "avg_duration": 0,
                "min_duration": 0,
                "max_duration": 0
            }
        
        return {
            "total_blinks": self.total_blinks,
            "avg_duration": sum(self.blink_durations) / len(self.blink_durations),
            "min_duration": min(self.blink_durations),
            "max_duration": max(self.blink_durations)
        }
    
    def reset_statistics(self):
        """
        Reset blink statistics.
        """
        self.total_blinks = 0
        self.blink_durations = []