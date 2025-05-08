"""
Alert System Module

This module handles alerts when fatigue is detected, including sound alerts
and visual notifications.
"""

import os
import time
import threading
import pygame
import cv2
import numpy as np

class AlertSystem:
    """
    Class for managing alerts when fatigue is detected.
    """
    def __init__(self, sound_file=None, cooldown=10):
        """
        Initialize the alert system.
        
        Args:
            sound_file (str, optional): Path to sound file for alerts
            cooldown (int): Cooldown period between alerts in seconds
        """
        self.sound_file = sound_file
        self.cooldown = cooldown
        
        # Initialize sound system
        self.sound_enabled = False
        if sound_file is not None and os.path.isfile(sound_file):
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(sound_file)
                self.sound_enabled = True
                print(f"Sound alert initialized with file: {sound_file}")
            except Exception as e:
                print(f"Error initializing sound alert: {e}")
        elif sound_file is not None:
            print(f"Warning: Sound file not found: {sound_file}")
            
            # Use default alert sound
            default_sound = os.path.join(os.path.dirname(__file__), "../data/alert.wav")
            if os.path.isfile(default_sound):
                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(default_sound)
                    self.sound_enabled = True
                    print(f"Using default sound alert: {default_sound}")
                except Exception as e:
                    print(f"Error initializing default sound alert: {e}")
        
        # Alert state
        self.last_alert_time = 0
        self.alert_active = False
        self.alert_thread = None
        
        # Visual alert settings
        self.visual_alert_frames = 0
        self.max_visual_alert_frames = 30  # Number of frames to show visual alert
        
        # Set up flash alert
        self.flash_color = (0, 0, 255)  # Red color
        self.flash_opacity = 0.3  # Initial opacity
        self.flash_increasing = True  # Whether opacity is increasing
    
    def trigger_alert(self):
        """
        Trigger an alert if cooldown period has passed.
        
        Returns:
            bool: True if alert was triggered, False otherwise
        """
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.cooldown:
            return False
        
        # Update alert time
        self.last_alert_time = current_time
        
        # Trigger sound alert
        if self.sound_enabled and not self.alert_active:
            self.alert_active = True
            
            # Start alert in a separate thread
            self.alert_thread = threading.Thread(target=self._play_alert)
            self.alert_thread.daemon = True
            self.alert_thread.start()
        
        # Reset visual alert
        self.visual_alert_frames = self.max_visual_alert_frames
        
        return True
    
    def _play_alert(self):
        """
        Play the alert sound.
        """
        try:
            pygame.mixer.music.play()
            
            # Wait for sound to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Reset alert state
            self.alert_active = False
            
        except Exception as e:
            print(f"Error playing alert sound: {e}")
            self.alert_active = False
    
    def apply_visual_alert(self, frame):
        """
        Apply a visual alert effect to the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with visual alert applied
        """
        # Check if visual alert is active
        if self.visual_alert_frames <= 0:
            return frame
        
        # Create a copy of the frame
        result = frame.copy()
        
        # Create overlay for flash effect
        overlay = result.copy()
        
        # Update flash opacity
        if self.flash_increasing:
            self.flash_opacity += 0.05
            if self.flash_opacity >= 0.7:
                self.flash_opacity = 0.7
                self.flash_increasing = False
        else:
            self.flash_opacity -= 0.05
            if self.flash_opacity <= 0.3:
                self.flash_opacity = 0.3
                self.flash_increasing = True
        
        # Apply colored overlay
        cv2.rectangle(overlay, (0, 0), (result.shape[1], result.shape[0]), self.flash_color, -1)
        cv2.addWeighted(overlay, self.flash_opacity, result, 1 - self.flash_opacity, 0, result)
        
        # Add "FATIGUE DETECTED" text
        cv2.putText(result, "FATIGUE DETECTED!", (result.shape[1]//2 - 150, result.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(result, "Please Take a Break!", (result.shape[1]//2 - 140, result.shape[0]//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Decrement frame counter
        self.visual_alert_frames -= 1
        
        return result
    
    def vibration_alert(self):
        """
        Trigger a vibration alert if hardware is available.
        
        Note: This is a placeholder for hardware integration.
        """
        print("Vibration alert triggered (hardware not connected)")
    
    def is_alert_active(self):
        """
        Check if an alert is currently active.
        
        Returns:
            bool: True if alert is active, False otherwise
        """
        return self.alert_active or self.visual_alert_frames > 0
    
    def set_cooldown(self, cooldown):
        """
        Set the cooldown period between alerts.
        
        Args:
            cooldown (int): Cooldown period in seconds
        """
        self.cooldown = cooldown
    
    def cleanup(self):
        """
        Clean up resources.
        """
        if self.sound_enabled:
            try:
                pygame.mixer.quit()
            except:
                pass