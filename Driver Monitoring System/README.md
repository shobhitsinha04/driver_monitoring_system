# Driver Fatigue Detection System

A real-time fatigue detection system using Python, OpenCV, MTCNN, and PyTorch, designed to monitor driver alertness and improve road safety.

## Features

- **Real-time face detection** using MTCNN (Multi-task Cascaded Convolutional Networks)
- **Custom deep learning model** built with PyTorch for eye state classification
- **Multi-factor fatigue detection** based on eye closure, blink rate, head position, and facial expressions
- **Alert system** with visual and audio notifications
- **OpenAI API integration** for intelligent feedback and monitoring
- **Optimized video processing pipeline** for improved real-time detection speed
- **Comprehensive logging and analytics** for session review

## Requirements

- Python 3.8+
- Webcam or camera device
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/driver-fatigue-detection.git
   cd driver-fatigue-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Set your OpenAI API key for enhanced feedback:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

1. Run the main application:
   ```
   python main.py
   ```

2. Additional command-line options:
   ```
   python main.py --camera 1  # Use an alternative camera (default: 0)
   python main.py --debug     # Enable debug mode with visualization
   python main.py --record    # Record the session
   python main.py --config custom_config.yaml  # Use a custom configuration
   ```

3. Controls:
   - Press 'q' to quit the application
   - Press 'f' to request feedback

## Configuration

The system can be configured using the `config.yaml` file. Key parameters include:

- Face detection sensitivity
- Eye aspect ratio thresholds
- Fatigue detection parameters
- Alert system settings
- OpenAI API settings

## Project Structure

```
driver_fatigue_detection/
│
├── main.py                   # Main application entry point
├── requirements.txt          # Project dependencies
├── config.yaml               # Configuration parameters
│
├── models/
│   ├── fatigue_detector.py   # PyTorch fatigue detection model
│   ├── eye_state_model.pt    # Pre-trained eye state model weights
│   └── face_detector.py      # MTCNN face detector wrapper
│
├── utils/
│   ├── video_processor.py    # Video processing pipeline
│   ├── landmarks.py          # Facial landmark detection utilities
│   ├── eye_analyzer.py       # Eye state analysis functions
│   └── alert_system.py       # Alert system for detected fatigue
│
├── feedback/
│   ├── openai_feedback.py    # OpenAI API integration for feedback
│   └── user_interaction.py   # User interaction module
```

## Model Training

The eye state classification model can be trained on custom data:

1. Prepare your dataset with 'open' and 'closed' eye images in the following structure:
   ```
   data/
   ├── train/
   │   ├── open/
   │   └── closed/
   └── test/
       ├── open/
       └── closed/
   ```

2. Train the model:
   ```python
   from models.fatigue_detector import FatigueDetector, EyeDataset
   from torch.utils.data import DataLoader
   from torchvision import transforms
   
   # Define transforms
   transform = transforms.Compose([
       transforms.Resize((64, 64)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5], std=[0.5])
   ])
   
   # Create datasets
   train_dataset = EyeDataset('data/train', transform=transform)
   val_dataset = EyeDataset('data/test', transform=transform)
   
   # Create data loaders
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
   
   # Initialize and train the model
   detector = FatigueDetector('models/eye_state_model.pt')
   history = detector.train(train_loader, val_loader, epochs=20, learning_rate=0.001)
   ```

## Performance

The system is designed to run efficiently in real-time:
- Target performance: 30+ FPS on modern hardware
- Detection accuracy: 90%+ for eye state classification
- Low latency alerts: <500ms from detection to notification

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [MTCNN](https://github.com/timesler/facenet-pytorch) for face detection
- [PyTorch](https://pytorch.org/) for deep learning framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [OpenAI](https://openai.com/) for intelligent feedback API