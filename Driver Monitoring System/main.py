#!/usr/bin/env python
"""
Driver Fatigue Detection System
Main application entry point

This module initializes and runs the real-time driver fatigue detection system,
integrating face detection, eye state analysis, and alert mechanisms.
"""

import os
import sys
import time
import argparse
import yaml
import cv2
import torch
import numpy as np
from datetime import datetime
import platform

# Import local modules
from models.face_detector import FaceDetector
from models.fatigue_detector import FatigueDetector
from utils.video_processor import VideoProcessor
from utils.eye_analyzer import EyeAnalyzer
from utils.alert_system import AlertSystem
from feedback.openai_feedback import OpenAIFeedback
from feedback.user_interaction import UserInteraction

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Driver Fatigue Detection System')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--record', action='store_true',
                        help='Record video output')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with visualization')
    parser.add_argument('--performance', type=str, default='balanced',
                    choices=['low', 'balanced', 'high'],
                    help='Performance mode (low, balanced, high)')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

class FatigueDetectionApp:
    """Main application class for driver fatigue detection."""
    
    def __init__(self, config, args):
        """Initialize the fatigue detection application."""
        print("Initializing Driver Fatigue Detection System...")
        
        # Configuration
        self.config = config
        self.args = args
        
        # Check for M2 performance mode
        if platform.processor() == 'arm' and 'apple' in platform.platform().lower():
            print("Detected Apple Silicon (M2). Optimizing performance...")
            if self.args.performance == 'low':
                # Lowest resource usage, but less accurate
                self.config['video']['resize_width'] = 320
                self.config['video']['resize_height'] = 240
                self.config['face_detection']['frame_skip'] = 4
            elif self.args.performance == 'balanced':
                # Balanced mode - default settings from config update above
                pass
            elif self.args.performance == 'high':
                # Highest accuracy, but more resource intensive
                self.config['video']['resize_width'] = 640
                self.config['video']['resize_height'] = 480
                self.config['face_detection']['frame_skip'] = 2
        
        # Set device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple M2 GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        # Initialize components
        self.init_components()
        
        # Performance metrics
        self.fps_list = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Fatigue metrics
        self.fatigue_score = 0
        self.consecutive_closed = 0
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.fatigue_detected = False
        
        # Recording setup
        self.out = None
        if args.record:
            self.setup_recording()
    
    def init_components(self):
        """Initialize all system components."""
        # Initialize face detector (MTCNN)
        self.face_detector = FaceDetector(
            min_face_size=self.config['face_detection']['min_face_size'],
            thresholds=self.config['face_detection']['thresholds'],
            device=self.device
        )
        
        # Initialize fatigue detector (PyTorch model)
        self.fatigue_detector = FatigueDetector(
            model_path=self.config['models']['eye_state_model'],
            device=self.device
        )
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            resize_width=self.config['video']['resize_width'],
            resize_height=self.config['video']['resize_height']
        )
        
        # Initialize eye analyzer
        self.eye_analyzer = EyeAnalyzer(
            eye_aspect_ratio_threshold=self.config['eye_analysis']['ear_threshold'],
            consecutive_frames=self.config['eye_analysis']['consecutive_frames']
        )
        
        # Initialize alert system
        self.alert_system = AlertSystem(
            sound_file=self.config['alerts']['sound_file'],
            cooldown=self.config['alerts']['cooldown']
        )
        
        # Initialize OpenAI feedback
        self.openai_feedback = OpenAIFeedback(
            api_key=os.environ.get('OPENAI_API_KEY') or self.config['openai']['api_key'],
            model=self.config['openai']['model']
        )
        
        # Initialize user interaction
        self.user_interaction = UserInteraction(
            feedback_module=self.openai_feedback
        )
    
    def setup_recording(self):
        """Set up video recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fatigue_detection_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Get camera dimensions
        cap = cv2.VideoCapture(self.args.camera)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        self.out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        print(f"Recording video to {filename}")
    
    def update_fatigue_metrics(self, eyes_closed, blink_detected):
        """Update fatigue metrics based on eye state."""
        # Update consecutive closed eyes counter
        if eyes_closed:
            self.consecutive_closed += 1
        else:
            self.consecutive_closed = 0
        
        # Update blink count
        if blink_detected:
            self.blink_count += 1
            current_time = time.time()
            time_diff = current_time - self.last_blink_time
            
            # If blink rate is too high, increase fatigue score
            if time_diff < self.config['fatigue_detection']['blink_time_threshold']:
                self.fatigue_score += 1
            
            self.last_blink_time = current_time
        
        # Check if fatigue is detected
        if self.consecutive_closed >= self.config['fatigue_detection']['consecutive_closed_threshold'] or \
           self.fatigue_score >= self.config['fatigue_detection']['fatigue_score_threshold']:
            if not self.fatigue_detected:
                self.fatigue_detected = True
                self.alert_system.trigger_alert()
                
                # Log fatigue detection for feedback
                self.user_interaction.log_fatigue_event({
                    'timestamp': datetime.now().isoformat(),
                    'consecutive_closed': self.consecutive_closed,
                    'fatigue_score': self.fatigue_score,
                    'blink_count': self.blink_count
                })
        else:
            self.fatigue_detected = False
        
        # Reset fatigue score periodically
        if time.time() - self.start_time > self.config['fatigue_detection']['reset_interval']:
            self.fatigue_score = max(0, self.fatigue_score - 1)  # Gradually decrease
            self.blink_count = 0
            self.start_time = time.time()
    
    def calculate_fps(self):
        """Calculate and return current FPS."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            fps = self.frame_count / elapsed_time
            self.fps_list.append(fps)
            self.frame_count = 0
            self.start_time = time.time()
            
            # Keep only the last 10 FPS values for averaging
            if len(self.fps_list) > 10:
                self.fps_list.pop(0)
            
            return sum(self.fps_list) / len(self.fps_list)
        
        return 0
    
    def display_info(self, frame, faces, eye_states):
        """Display information on the frame for debugging."""
        # Draw face bounding boxes
        for face_idx, face in enumerate(faces):
            # Extract bounding box
            x1, y1, x2, y2 = [int(coord) for coord in face['box']]
            
            # Draw rectangle around face
            color = (0, 255, 0) if not self.fatigue_detected else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw eye landmarks
            left_eye = face['keypoints']['left_eye']
            right_eye = face['keypoints']['right_eye']
            cv2.circle(frame, (int(left_eye[0]), int(left_eye[1])), 2, (255, 0, 0), 2)
            cv2.circle(frame, (int(right_eye[0]), int(right_eye[1])), 2, (255, 0, 0), 2)
            
            # Display eye state
            eye_state = eye_states[face_idx] if face_idx < len(eye_states) else "Unknown"
            eye_text = f"Eyes: {eye_state}"
            cv2.putText(frame, eye_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display fatigue metrics
        cv2.putText(frame, f"Fatigue Score: {self.fatigue_score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Consecutive Closed: {self.consecutive_closed}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display FPS
        fps = self.calculate_fps()
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display fatigue warning
        if self.fatigue_detected:
            cv2.putText(frame, "FATIGUE DETECTED!", (frame.shape[1]//2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame for fatigue detection."""
        # Preprocess frame
        processed_frame = self.video_processor.preprocess(frame)
        
        # Detect faces
        faces = self.face_detector.detect_faces(processed_frame)
        
        if not faces:
            return frame, [], []
        
        # Extract eye regions and detect fatigue
        eye_states = []
        blink_detected = False
        
        for face in faces:
            # Get eye regions
            left_eye_region, right_eye_region = self.eye_analyzer.extract_eye_regions(processed_frame, face)
            
            # Skip if eyes not properly detected
            if left_eye_region is None or right_eye_region is None:
                eye_states.append("Unknown")
                continue
            
            # Check eye aspect ratio for blink detection
            left_ear = self.eye_analyzer.calculate_eye_aspect_ratio(left_eye_region)
            right_ear = self.eye_analyzer.calculate_eye_aspect_ratio(right_eye_region)
            avg_ear = (left_ear + right_ear) / 2
            
            # Use deep learning model for eye state classification
            eye_state = self.fatigue_detector.predict_eye_state(left_eye_region, right_eye_region)
            eye_states.append(eye_state)
            
            # Determine if eyes are closed based on both model and EAR
            eyes_closed = (eye_state == "Closed" or 
                           avg_ear < self.config['eye_analysis']['ear_threshold'])
            
            # Detect blink
            if self.eye_analyzer.is_blink(avg_ear):
                blink_detected = True
            
            # Update fatigue metrics
            self.update_fatigue_metrics(eyes_closed, blink_detected)
        
        # Display information on frame if debug mode is enabled
        if self.args.debug:
            frame = self.display_info(frame, faces, eye_states)
        
        return frame, faces, eye_states
    
    def run(self):
        """Run the fatigue detection application."""
        # Open video capture
        cap = cv2.VideoCapture(self.args.camera)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.args.camera}")
            return
        
        print("Driver Fatigue Detection System running. Press 'q' to quit.")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, faces, eye_states = self.process_frame(frame)
                
                # Record if enabled
                if self.args.record and self.out is not None:
                    self.out.write(processed_frame)
                
                # Display frame
                cv2.imshow('Driver Fatigue Detection', processed_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Check for feedback request
                if key == ord('f'):
                    feedback = self.user_interaction.request_feedback()
                    print(f"Feedback: {feedback}")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Clean up
            cap.release()
            if self.args.record and self.out is not None:
                self.out.release()
            cv2.destroyAllWindows()
            
            # Generate session summary with OpenAI
            if self.config['openai']['generate_summary']:
                summary = self.user_interaction.generate_session_summary()
                print("\nSession Summary:")
                print(summary)

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize and run the application
    app = FatigueDetectionApp(config, args)
    app.run()

if __name__ == "__main__":
    main()