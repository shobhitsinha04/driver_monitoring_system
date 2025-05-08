"""
Video Processor Module

This module handles video processing tasks for the fatigue detection system,
including frame acquisition, preprocessing, and optimization.
"""

import cv2
import numpy as np
import threading
import queue
import time
import platform

class VideoProcessor:
    """
    Class for processing video frames from a camera or file.
    """
    def __init__(self, resize_width=640, resize_height=480, 
                 buffer_size=5, enable_threading=True):
        """
        Initialize the video processor.
        
        Args:
            resize_width (int): Width to resize frames to
            resize_height (int): Height to resize frames to
            buffer_size (int): Size of the frame buffer for threading
            enable_threading (bool): Whether to enable threaded processing
        """
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.buffer_size = buffer_size
        self.enable_threading = enable_threading
        
        # Frame buffer for threaded processing
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        
        # Threading setup
        self.is_running = False
        self.capture_thread = None
        
        # Performance metrics
        self.fps = 0
        self.processing_times = []
    
    def start_capture(self, camera_id=0):
        """
        Start capturing frames from the camera.
        
        Args:
            camera_id (int): Camera device ID
            
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Start threaded capture if enabled
        if self.enable_threading:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._threaded_capture)
            self.capture_thread.daemon = True
            self.capture_thread.start()
        
        return True
    
    def _threaded_capture(self):
        """
        Thread function for capturing frames.
        """
        while self.is_running:
            if not self.frame_buffer.full():
                ret, frame = self.camera.read()
                
                if ret:
                    # Preprocess frame
                    processed_frame = self.preprocess(frame)
                    
                    # Add to buffer with timestamp
                    self.frame_buffer.put((processed_frame, time.time()))
                else:
                    print("Error: Failed to grab frame")
                    self.is_running = False
                    break
            else:
                # Small sleep to prevent CPU hogging when buffer is full
                time.sleep(0.001)
    
    def get_frame(self):
        """
        Get the next frame from the camera or buffer.
        
        Returns:
            numpy.ndarray: Processed frame
        """
        if self.enable_threading:
            if not self.frame_buffer.empty():
                frame, timestamp = self.frame_buffer.get()
                latency = time.time() - timestamp
                return frame
            else:
                return None
        else:
            ret, frame = self.camera.read()
            
            if ret:
                return self.preprocess(frame)
            else:
                return None
    
    def preprocess(self, frame):
        """
        Preprocess a frame for analysis.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        start_time = time.time()
        
        # Resize the frame
        if frame.shape[1] != self.resize_width or frame.shape[0] != self.resize_height:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        
        # For M2 Macs, skip expensive image enhancements if FPS is below target
        if len(self.processing_times) > 10:
            avg_fps = len(self.processing_times) / sum(self.processing_times)
            if avg_fps < 15 and platform.processor() == 'arm' and 'apple' in platform.platform().lower():
                # Just apply basic contrast normalization
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                
                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Keep only the last 100 times
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                # Update FPS
                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    self.fps = 1.0 / avg_time if avg_time > 0 else 0
                
                return frame
        
        # Apply basic image enhancements
        # 1. Normalize brightness and contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only the last 100 times
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        # Update FPS
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            self.fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return enhanced_frame
    
    def apply_roi(self, frame, roi=None):
        """
        Apply region of interest (ROI) to a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            roi (tuple, optional): ROI coordinates (x, y, width, height)
            
        Returns:
            numpy.ndarray: Frame with ROI applied
        """
        if roi is None:
            # Default ROI is middle 80% of the frame
            height, width = frame.shape[:2]
            margin_x = int(width * 0.1)
            margin_y = int(height * 0.1)
            roi = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
        
        x, y, w, h = roi
        return frame[y:y+h, x:x+w]
    
    def denoise_frame(self, frame, strength=10):
        """
        Apply denoising to a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            strength (int): Denoising strength
            
        Returns:
            numpy.ndarray: Denoised frame
        """
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)
    
    def draw_fps(self, frame):
        """
        Draw FPS on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with FPS drawn
        """
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
    
    def stop_capture(self):
        """
        Stop capturing frames.
        """
        self.is_running = False
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1)
        
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
    
    def get_performance_metrics(self):
        """
        Get performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return {
            "fps": self.fps,
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            "min_processing_time": min(self.processing_times) if self.processing_times else 0,
            "max_processing_time": max(self.processing_times) if self.processing_times else 0,
        }

class AsyncVideoProcessor(VideoProcessor):
    """
    Extended VideoProcessor with asynchronous processing capabilities.
    """
    def __init__(self, resize_width=640, resize_height=480, buffer_size=5):
        """
        Initialize the asynchronous video processor.
        
        Args:
            resize_width (int): Width to resize frames to
            resize_height (int): Height to resize frames to
            buffer_size (int): Size of the frame buffer
        """
        super().__init__(resize_width, resize_height, buffer_size, True)
        
        # Additional buffer for processed frames
        self.processed_frames = queue.Queue(maxsize=buffer_size)
        
        # Processing thread
        self.processing_thread = None
    
    def start_processing(self, processor_func):
        """
        Start asynchronous frame processing.
        
        Args:
            processor_func (callable): Function to process frames
            
        Returns:
            bool: True if processing started successfully, False otherwise
        """
        if not self.is_running:
            print("Error: Capture not started")
            return False
        
        self.processor_func = processor_func
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
    
    def _processing_loop(self):
        """
        Thread function for processing frames.
        """
        while self.is_running:
            if not self.frame_buffer.empty() and not self.processed_frames.full():
                # Get frame from buffer
                frame, timestamp = self.frame_buffer.get()
                
                # Process frame
                try:
                    result = self.processor_func(frame)
                    
                    # Add to processed frames buffer
                    self.processed_frames.put((result, frame, timestamp))
                except Exception as e:
                    print(f"Error processing frame: {e}")
            else:
                # Small sleep to prevent CPU hogging
                time.sleep(0.001)
    
    def get_processed_frame(self):
        """
        Get the next processed frame.
        
        Returns:
            tuple: (result, original_frame, latency)
        """
        if not self.processed_frames.empty():
            result, original_frame, timestamp = self.processed_frames.get()
            latency = time.time() - timestamp
            return result, original_frame, latency
        else:
            return None, None, 0
    
    def stop(self):
        """
        Stop capturing and processing.
        """
        self.is_running = False
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1)
        
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1)
        
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()