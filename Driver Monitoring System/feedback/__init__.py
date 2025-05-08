"""
Feedback package for the Driver Fatigue Detection System.

This package contains modules for OpenAI API integration and user interaction.
"""

from .openai_feedback import OpenAIFeedback
from .user_interaction import UserInteraction

__all__ = ['OpenAIFeedback', 'UserInteraction']