"""
User Interaction Module

This module handles interactions with the user, providing feedback and collecting responses.
"""

import time
import threading
from datetime import datetime

class UserInteraction:
    """
    Class for managing user interactions and feedback.
    """
    def __init__(self, feedback_module, voice_enabled=True):
        """
        Initialize the user interaction module.
        
        Args:
            feedback_module: Module for generating feedback (e.g., OpenAIFeedback)
            voice_enabled (bool): Whether to enable voice feedback
        """
        self.feedback_module = feedback_module
        self.voice_enabled = voice_enabled
        
        # Initialize text-to-speech if enabled
        self.tts_engine = None
        if voice_enabled:
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)  # Speaking rate
                self.tts_engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
                print("Text-to-speech initialized")
            except ImportError:
                print("pyttsx3 not installed. Voice feedback disabled.")
                self.voice_enabled = False
            except Exception as e:
                print(f"Error initializing text-to-speech: {e}")
                self.voice_enabled = False
        
        # Keep track of interactions
        self.interaction_history = []
        
        # Active feedback management
        self.last_feedback_time = 0
        self.feedback_cooldown = 10  # seconds
        self.feedback_thread = None
        self.stop_feedback = threading.Event()
    
    def log_fatigue_event(self, event_data):
        """
        Log a fatigue event and pass it to the feedback module.
        
        Args:
            event_data (dict): Event data
        """
        # Log the event
        self.feedback_module.log_fatigue_event(event_data)
        
        # Check if feedback should be triggered
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            self.last_feedback_time = current_time
            
            # Determine fatigue level
            if event_data.get('consecutive_closed', 0) >= 5 or event_data.get('fatigue_score', 0) >= 8:
                self.provide_feedback("high_fatigue")
            elif event_data.get('consecutive_closed', 0) >= 3 or event_data.get('fatigue_score', 0) >= 5:
                self.provide_feedback("moderate_fatigue")
    
    def provide_feedback(self, fatigue_level):
        """
        Provide feedback to the user based on fatigue level.
        
        Args:
            fatigue_level (str): Level of fatigue
        """
        # Generate feedback using the feedback module
        feedback = self.feedback_module.generate_feedback(fatigue_level)
        
        # Log the interaction
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "system_feedback",
            "fatigue_level": fatigue_level,
            "content": feedback
        })
        
        # Display feedback on console
        print(f"FEEDBACK: {feedback}")
        
        # Provide voice feedback if enabled
        if self.voice_enabled and self.tts_engine is not None:
            # Run in a separate thread to avoid blocking the main thread
            if self.feedback_thread is not None and self.feedback_thread.is_alive():
                self.stop_feedback.set()
                self.feedback_thread.join(timeout=1)
                self.stop_feedback.clear()
            
            self.feedback_thread = threading.Thread(
                target=self._speak_feedback,
                args=(feedback,)
            )
            self.feedback_thread.daemon = True
            self.feedback_thread.start()
        
        return feedback
    
    def _speak_feedback(self, feedback):
        """
        Speak feedback using text-to-speech.
        
        Args:
            feedback (str): Feedback text to speak
        """
        try:
            self.tts_engine.say(feedback)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error speaking feedback: {e}")
    
    def request_feedback(self):
        """
        Request feedback from the feedback module.
        
        Returns:
            str: Generated feedback
        """
        # Analyze driving behavior
        analysis = self.feedback_module.analyze_driving_behavior()
        
        # Generate appropriate feedback based on severity
        if "severity" in analysis:
            if analysis["severity"] == "high":
                feedback = self.provide_feedback("high_fatigue")
            elif analysis["severity"] == "moderate":
                feedback = self.provide_feedback("moderate_fatigue")
            else:
                feedback = self.provide_feedback("preventive")
        else:
            feedback = self.provide_feedback("preventive")
        
        return feedback
    
    def generate_session_summary(self):
        """
        Generate a summary of the current session.
        
        Returns:
            str: Session summary
        """
        return self.feedback_module.generate_session_summary()
    
    def save_session_data(self, filename=None):
        """
        Save session data to a file.
        
        Args:
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved file
        """
        return self.feedback_module.save_session_data(filename)
    
    def get_interaction_history(self):
        """
        Get the interaction history.
        
        Returns:
            list: Interaction history
        """
        return self.interaction_history
    
    def cleanup(self):
        """
        Clean up resources.
        """
        if self.voice_enabled and self.tts_engine is not None:
            if self.feedback_thread is not None and self.feedback_thread.is_alive():
                self.stop_feedback.set()
                self.feedback_thread.join(timeout=1)
            
            try:
                self.tts_engine.stop()
            except:
                pass