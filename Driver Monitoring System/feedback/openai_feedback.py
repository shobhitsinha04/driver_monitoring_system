"""
OpenAI Feedback Module

This module implements integration with OpenAI APIs for user feedback and monitoring,
enhancing overall user safety and interaction efficiency.
"""

import os
import json
import time
import requests
from datetime import datetime

class OpenAIFeedback:
    """
    Class for OpenAI API integration to provide intelligent feedback and monitoring.
    """
    def __init__(self, api_key=None, model="gpt-4"):
        """
        Initialize the OpenAI feedback module.
        
        Args:
            api_key (str, optional): OpenAI API key
            model (str): OpenAI model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OpenAI API key not provided. Feedback features will be disabled.")
        
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Session data for feedback
        self.session_data = {
            "start_time": datetime.now().isoformat(),
            "fatigue_events": [],
            "user_interactions": [],
            "feedback_provided": []
        }
        
        # Initialize feedback strategies
        self.feedback_strategies = {
            "high_fatigue": [
                "I've noticed you're showing signs of fatigue. Would you like to take a break?",
                "Your eyes appear to be closing frequently. Consider stopping to rest.",
                "For your safety, I recommend pulling over and taking a 20-minute nap.",
                "Fatigue detected! Please consider stopping at the next rest area."
            ],
            "moderate_fatigue": [
                "You're showing early signs of fatigue. Would you like some tips to stay alert?",
                "I've detected some tiredness indicators. Consider opening a window for fresh air.",
                "Your blink rate has increased. Try stretching or adjusting your position."
            ],
            "preventive": [
                "Remember to take regular breaks on long journeys.",
                "Staying hydrated can help maintain alertness while driving.",
                "If you're feeling tired, caffeine can provide a temporary boost, but isn't a substitute for rest."
            ]
        }
    
    def log_fatigue_event(self, event_data):
        """
        Log a fatigue detection event.
        
        Args:
            event_data (dict): Event data
        """
        self.session_data["fatigue_events"].append(event_data)
    
    def log_user_interaction(self, interaction_data):
        """
        Log a user interaction.
        
        Args:
            interaction_data (dict): Interaction data
        """
        self.session_data["user_interactions"].append(interaction_data)
    
    def log_feedback(self, feedback_data):
        """
        Log feedback provided to the user.
        
        Args:
            feedback_data (dict): Feedback data
        """
        self.session_data["feedback_provided"].append(feedback_data)
    
    def generate_feedback(self, fatigue_level="high_fatigue"):
        """
        Generate feedback based on the current state.
        
        Args:
            fatigue_level (str): Level of fatigue
            
        Returns:
            str: Generated feedback
        """
        # If API key is not available, use predefined feedback
        if not self.api_key:
            import random
            return random.choice(self.feedback_strategies[fatigue_level])
        
        # Prepare recent events for context
        recent_events = self.session_data["fatigue_events"][-5:] if self.session_data["fatigue_events"] else []
        recent_feedback = self.session_data["feedback_provided"][-3:] if self.session_data["feedback_provided"] else []
        
        # Create prompt for OpenAI
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant integrated into a driver fatigue detection system. "
                    "Your role is to provide helpful, supportive feedback to drivers who may be "
                    "experiencing fatigue. Your feedback should be concise (1-2 sentences), "
                    "respectful, and safety-focused. Avoid being alarmist, but be clear about "
                    "safety concerns when appropriate."
                )
            },
            {
                "role": "user",
                "content": f"""
                Generate appropriate feedback for a driver with the following fatigue level: {fatigue_level}.
                
                Recent fatigue events: {json.dumps(recent_events, indent=2)}
                
                Previous feedback given: {json.dumps(recent_feedback, indent=2)}
                
                The feedback should be personalized based on the data, supportive, and focus on safety.
                Keep it concise - no more than 1-2 sentences.
                """
            }
        ]
        
        try:
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                timeout=5  # 5-second timeout for real-time feedback
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Extract and return feedback
            feedback = response.json()["choices"][0]["message"]["content"].strip()
            
            # Log the provided feedback
            self.log_feedback({
                "timestamp": datetime.now().isoformat(),
                "fatigue_level": fatigue_level,
                "feedback": feedback
            })
            
            return feedback
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            
            # Fall back to predefined feedback
            import random
            feedback = random.choice(self.feedback_strategies[fatigue_level])
            
            # Log the provided feedback
            self.log_feedback({
                "timestamp": datetime.now().isoformat(),
                "fatigue_level": fatigue_level,
                "feedback": feedback,
                "error": str(e)
            })
            
            return feedback
    
    def analyze_driving_behavior(self):
        """
        Analyze driving behavior and fatigue patterns.
        
        Returns:
            dict: Analysis results
        """
        if not self.session_data["fatigue_events"]:
            return {
                "message": "Insufficient data for analysis.",
                "recommendation": "Continue monitoring driver behavior."
            }
        
        # Basic analysis without API
        if not self.api_key:
            # Count fatigue events
            event_count = len(self.session_data["fatigue_events"])
            
            # Basic analysis
            if event_count > 10:
                severity = "high"
                recommendation = "The driver shows significant signs of fatigue. Recommend stopping for rest."
            elif event_count > 5:
                severity = "moderate"
                recommendation = "The driver shows moderate signs of fatigue. Recommend increased vigilance."
            else:
                severity = "low"
                recommendation = "The driver shows minimal signs of fatigue. Continue monitoring."
            
            return {
                "event_count": event_count,
                "severity": severity,
                "recommendation": recommendation
            }
        
        # Advanced analysis with OpenAI
        try:
            # Create prompt for analysis
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant specialized in analyzing driver fatigue patterns. "
                        "Based on the data provided, analyze the driver's fatigue patterns and provide "
                        "actionable insights and recommendations. Focus on safety and practical advice."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze the following driver fatigue data:
                    
                    Session start time: {self.session_data["start_time"]}
                    Current time: {datetime.now().isoformat()}
                    
                    Fatigue events: {json.dumps(self.session_data["fatigue_events"], indent=2)}
                    
                    User interactions: {json.dumps(self.session_data["user_interactions"], indent=2)}
                    
                    Provide a concise analysis with:
                    1. A severity assessment (low, moderate, high)
                    2. Key patterns observed
                    3. A practical recommendation
                    
                    Format the response as JSON with the following fields:
                    - severity
                    - patterns (array)
                    - recommendation
                    """
                }
            ]
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.3
                },
                timeout=10
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Extract analysis
            analysis_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response (handling possible formatting issues)
            try:
                # Try to extract JSON from the response
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # If not valid JSON, extract fields manually
                import re
                
                severity_match = re.search(r'"severity":\s*"([^"]+)"', analysis_text)
                severity = severity_match.group(1) if severity_match else "unknown"
                
                recommendation_match = re.search(r'"recommendation":\s*"([^"]+)"', analysis_text)
                recommendation = recommendation_match.group(1) if recommendation_match else "No specific recommendation."
                
                patterns_match = re.search(r'"patterns":\s*\[(.*?)\]', analysis_text, re.DOTALL)
                patterns_text = patterns_match.group(1) if patterns_match else ""
                patterns = [p.strip().strip('"') for p in patterns_text.split(",") if p.strip()]
                
                analysis = {
                    "severity": severity,
                    "patterns": patterns,
                    "recommendation": recommendation
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing driving behavior: {e}")
            
            # Fall back to basic analysis
            event_count = len(self.session_data["fatigue_events"])
            
            if event_count > 10:
                severity = "high"
                recommendation = "The driver shows significant signs of fatigue. Recommend stopping for rest."
            elif event_count > 5:
                severity = "moderate"
                recommendation = "The driver shows moderate signs of fatigue. Recommend increased vigilance."
            else:
                severity = "low"
                recommendation = "The driver shows minimal signs of fatigue. Continue monitoring."
            
            return {
                "event_count": event_count,
                "severity": severity,
                "recommendation": recommendation,
                "error": str(e)
            }
    
    def generate_session_summary(self):
        """
        Generate a summary of the current session.
        
        Returns:
            str: Session summary
        """
        # If no API key or no events, return basic summary
        if not self.api_key or not self.session_data["fatigue_events"]:
            session_duration = (datetime.now() - datetime.fromisoformat(self.session_data["start_time"])).total_seconds() / 60
            event_count = len(self.session_data["fatigue_events"])
            
            return (
                f"Session Summary:\n"
                f"- Duration: {session_duration:.1f} minutes\n"
                f"- Fatigue events detected: {event_count}\n"
                f"- Overall assessment: {'Fatigue detected' if event_count > 0 else 'No fatigue detected'}"
            )
        
        try:
            # Create prompt for summary
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant integrated into a driver fatigue detection system. "
                        "Your task is to generate a comprehensive summary of the driving session, "
                        "focusing on fatigue events, patterns, and safety recommendations."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                    Generate a session summary based on the following data:
                    
                    Session start time: {self.session_data["start_time"]}
                    Current time: {datetime.now().isoformat()}
                    
                    Fatigue events: {json.dumps(self.session_data["fatigue_events"], indent=2)}
                    
                    User interactions: {json.dumps(self.session_data["user_interactions"], indent=2)}
                    
                    Feedback provided: {json.dumps(self.session_data["feedback_provided"], indent=2)}
                    
                    The summary should be concise but informative, focusing on:
                    1. Overall fatigue assessment
                    2. Key patterns observed
                    3. Safety recommendations
                    4. Actionable insights for future driving sessions
                    """
                }
            ]
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.5
                }
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Extract and return summary
            summary = response.json()["choices"][0]["message"]["content"].strip()
            return summary
            
        except Exception as e:
            print(f"Error generating session summary: {e}")
            
            # Fall back to basic summary
            session_duration = (datetime.now() - datetime.fromisoformat(self.session_data["start_time"])).total_seconds() / 60
            event_count = len(self.session_data["fatigue_events"])
            
            return (
                f"Session Summary:\n"
                f"- Duration: {session_duration:.1f} minutes\n"
                f"- Fatigue events detected: {event_count}\n"
                f"- Overall assessment: {'Fatigue detected' if event_count > 0 else 'No fatigue detected'}\n"
                f"Note: Detailed analysis unavailable due to API error: {e}"
            )
    
    def save_session_data(self, filename=None):
        """
        Save session data to a file.
        
        Args:
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fatigue_session_{timestamp}.json"
        
        # Add end time
        self.session_data["end_time"] = datetime.now().isoformat()
        
        # Calculate session duration
        start_time = datetime.fromisoformat(self.session_data["start_time"])
        end_time = datetime.fromisoformat(self.session_data["end_time"])
        duration_seconds = (end_time - start_time).total_seconds()
        
        self.session_data["duration_seconds"] = duration_seconds
        
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            print(f"Session data saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving session data: {e}")
            return None