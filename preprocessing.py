# preprocessing.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Change image size to match CNN expectation
IMG_SIZE = (224, 224)  # Update to match model's first convolutional layer

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
                    print(f"Resized image shape: {img.shape}")
                    images.append(img)
                    labels.append(label_folder)  # Assuming each subfolder name is a label

    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0  # Normalize images
    images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)  # Change shape dynamically

    # Convert labels to integers
    unique_labels = sorted(set(labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_int[label] for label in labels])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)   
    return X_train, X_test, y_train, y_test

# Test script
if __name__ == "__main__":
    images, labels = load_images_from_folder()
    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
