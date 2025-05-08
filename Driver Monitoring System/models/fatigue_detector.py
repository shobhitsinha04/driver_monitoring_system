"""
Fatigue Detector Module

This module implements a PyTorch-based deep learning model for fatigue detection
through eye state classification.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

class EyeStateClassifier(nn.Module):
    """
    PyTorch CNN model for eye state classification.
    Classifies eye images as "Open" or "Closed".
    """
    def __init__(self):
        super(EyeStateClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: open, closed
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # Input shape: [batch_size, 1, 64, 64] (grayscale eye images)
        
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [batch_size, 32, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [batch_size, 64, 16, 16]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [batch_size, 128, 8, 8]
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class FatigueDetector:
    """
    Class for detecting driver fatigue using a trained eye state classifier.
    """
    def __init__(self, model_path, device=None):
        """
        Initialize the fatigue detector.
        
        Args:
            model_path (str): Path to the trained PyTorch model
            device (torch.device, optional): Device to run the model on
        """
        # Check if MPS (Apple Silicon GPU) is available
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple M2 GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        else:
            self.device = device
        
        # Initialize the model
        self.model = EyeStateClassifier()
        
        # Load the pre-trained model if it exists
        # Load the pre-trained model if it exists
        try:
            if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {model_path}")
            else:
                print(f"Warning: Model file {model_path} not found or empty. Using untrained model.")
                # Save an untrained model
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print(f"Created new untrained model at {model_path}")
        except (EOFError, RuntimeError, Exception) as e:
            print(f"Error loading model: {e}")
            print("Creating new untrained model...")
            # Remove corrupted file if it exists
            if os.path.isfile(model_path):
                try:
                    os.remove(model_path)
                    print(f"Removed corrupted model file: {model_path}")
                except:
                    print(f"Warning: Could not remove corrupted model file: {model_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save a new untrained model
            torch.save(self.model.state_dict(), model_path)
            print(f"Created new untrained model at {model_path}")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def preprocess_eye_image(self, eye_img):
        """
        Preprocess the eye image for the model.
        """
        # Convert to grayscale
        if len(eye_img.shape) == 3:
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        try:
            # Resize to 64x64
            eye_img = cv2.resize(eye_img, (64, 64))
            
            # Apply more advanced preprocessing
            # Enhance contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            eye_img = clahe.apply(eye_img)
            
            # Apply Gaussian blur to reduce noise
            eye_img = cv2.GaussianBlur(eye_img, (3, 3), 0)
            
            # Convert to tensor and normalize
            eye_tensor = torch.from_numpy(eye_img).float().unsqueeze(0) / 255.0
            eye_tensor = (eye_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Add batch dimension
            eye_tensor = eye_tensor.unsqueeze(0)
            
            return eye_tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing eye image: {e}")
            return None
    
    def predict_single_eye(self, eye_img):
        """
        Predict the state of a single eye.
        
        Args:
            eye_img (numpy.ndarray): Eye image
            
        Returns:
            str: "Open" or "Closed"
        """
        # Preprocess the eye image
        eye_tensor = self.preprocess_eye_image(eye_img)
        
        if eye_tensor is None:
            return "Unknown"
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(eye_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Convert prediction to class label
            state = "Open" if predicted.item() == 0 else "Closed"
            
        return state
    
    def predict_eye_state(self, left_eye, right_eye):
        """
        Predict the state of both eyes and determine overall eye state.
        """
        # Fallback to basic analysis if eye images are problematic
        if left_eye is None or right_eye is None or left_eye.size == 0 or right_eye.size == 0:
            return "Unknown"
        
        # Basic size check
        if left_eye.shape[0] < 10 or left_eye.shape[1] < 10 or right_eye.shape[0] < 10 or right_eye.shape[1] < 10:
            return "Unknown"
            
        # Calculate basic eye aspect ratio as fallback
        import cv2
        import numpy as np
        
        def calculate_simple_ear(eye_img):
            if eye_img is None or len(eye_img.shape) < 2:
                return 1.0
            h, w = eye_img.shape[:2]
            return h / w if w > 0 else 1.0
        
        left_ear = calculate_simple_ear(left_eye)
        right_ear = calculate_simple_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        
        # Predict using model
        left_state = self.predict_single_eye(left_eye)
        right_state = self.predict_single_eye(right_eye)
        
        # Handle unknown states
        if left_state == "Unknown" or right_state == "Unknown":
            # Fall back to EAR-based detection
            return "Closed" if avg_ear < 0.2 else "Open"
        
        # Combine model predictions with EAR for more robust detection
        model_says_closed = (left_state == "Closed" and right_state == "Closed")
        ear_says_closed = avg_ear < 0.2
        
        if model_says_closed or ear_says_closed:
            return "Closed"
        elif left_state == "Open" and right_state == "Open" and avg_ear >= 0.3:
            return "Open"
        else:
            return "Partially Closed"
        
    def train(self, train_loader, val_loader=None, epochs=10, learning_rate=0.001, save_path=None):
        """
        Train the eye state classifier model.
        
        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader
            val_loader (torch.utils.data.DataLoader, optional): Validation data loader
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            save_path (str, optional): Path to save the trained model
            
        Returns:
            dict: Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(epochs):
            # Training
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Learning rate scheduler step
                scheduler.step(val_loss)
                
                # Save best model
                if val_acc > best_val_acc and save_path:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Save model
                if save_path and epoch == epochs - 1:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
        
        # Set model back to evaluation mode
        self.model.eval()
        
        return history
    
    def evaluate(self, data_loader, criterion=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader (torch.utils.data.DataLoader): Data loader
            criterion (torch.nn.Module, optional): Loss function
            
        Returns:
            tuple: (loss, accuracy)
        """
        # Store current training mode
        was_training = self.model.training
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Default criterion
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        # Set model back to previous mode
        self.model.train(was_training)
        
        return loss, accuracy

# Create a dataset class for eye images
class EyeDataset(torch.utils.data.Dataset):
    """
    Dataset class for eye images.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get image paths and labels
        self.data = []
        
        # Open eyes
        open_dir = os.path.join(data_dir, 'open')
        if os.path.isdir(open_dir):
            for filename in os.listdir(open_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.data.append((os.path.join(open_dir, filename), 0))
        
        # Closed eyes
        closed_dir = os.path.join(data_dir, 'closed')
        if os.path.isdir(closed_dir):
            for filename in os.listdir(closed_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.data.append((os.path.join(closed_dir, filename), 1))
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, label) where label is 0 for open and 1 for closed
        """
        img_path, label = self.data[idx]
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = cv2.resize(image, (64, 64))
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        
        return image, label