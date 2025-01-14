import os
import shutil
from sklearn.model_selection import train_test_split
import zipfile

def setup_dataset():
    # Create directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # Create directories if they don't exist
    for split in ['train', 'val']:
        for state in ['open', 'closed']:
            os.makedirs(os.path.join(dataset_dir, split, state), exist_ok=True)
    
    # Path to downloaded zip file
    zip_path = os.path.join(base_dir, 'mrlEyes_2018_01.zip')
    
    # Check if zip file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip file not found at {zip_path}. Please download it first.")
    
    # Extract zip file
    print("Extracting dataset...")
    temp_dir = os.path.join(base_dir, 'temp_dataset')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Collect and organize images
    images = []
    labels = []
    
    print("Organizing dataset...")
    # Walk through all subdirectories
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                
                # New pattern: s0016_00083_1_0_0_0_1_01.png
                # The 6th number (0 or 1) indicates eye state
                parts = file.split('_')
                if len(parts) >= 7:
                    eye_state = parts[5]  # Get the 6th number
                    if eye_state == '1':
                        labels.append('closed')
                        images.append(file_path)
                    elif eye_state == '0':
                        labels.append('open')
                        images.append(file_path)
    
    # Check if we found any images
    if not images:
        raise ValueError("No images found in the dataset. Check if the ZIP file is correctly structured.")
    
    print(f"\nFound {len(images)} total images")
    print(f"Open eyes: {labels.count('open')}")
    print(f"Closed eyes: {labels.count('closed')}")
    
    # Split into train/val sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Copy files to appropriate directories
    def copy_files(files, labels, split):
        for file, label in zip(files, labels):
            dest_dir = os.path.join(dataset_dir, split, label)
            shutil.copy2(file, os.path.join(dest_dir, os.path.basename(file)))
    
    print("\nCopying files to train set...")
    copy_files(X_train, y_train, 'train')
    
    print("Copying files to validation set...")
    copy_files(X_val, y_val, 'val')
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    print("\nDataset setup complete!")
    
    # Print dataset statistics
    print("\nFinal Dataset Statistics:")
    for split in ['train', 'val']:
        for state in ['open', 'closed']:
            path = os.path.join(dataset_dir, split, state)
            count = len(os.listdir(path))
            print(f"{split}/{state}: {count} images")

if __name__ == "__main__":
    try:
        setup_dataset()
    except Exception as e:
        print(f"Error: {str(e)}")