import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Image size to match CNN input
IMG_SIZE = (224, 224)

# Dataset folder path
folder_path = "./DSETS/train"

def load_images_from_folder(img_size=IMG_SIZE):  
    images = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        
        if os.path.isdir(label_path):
            for img_filename in os.listdir(label_path):
                img_path = os.path.join(label_path, img_filename)
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize to 224x224
                    images.append(img)
                    labels.append(label_folder)  # Assign folder name as label

    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0  # Normalize images
    images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)  # Reshape for CNN input

    # Convert labels to integer encoding
    unique_labels = sorted(set(labels))  # Ensure consistent label ordering
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_int[label] for label in labels])

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)   
    
    return X_train, X_test, y_train, y_test, label_to_int  # Return mapping as well

# Run preprocessing if script is executed directly
if __name__ == "__main__":
    images, labels = load_images_from_folder()
    print(f"Loaded {len(images)} images from {len(set(labels))} categories: {set(labels)}")
    
    X_train, X_test, y_train, y_test, label_map = preprocess_data(images, labels)
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"Label Mapping: {label_map}")
