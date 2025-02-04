# preprocessing.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

folder_path = "./DSETS/train"

def load_images_from_folder(img_size=(224, 224)):
    images = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        
        if os.path.isdir(label_path):
            for img_filename in os.listdir(label_path):
                img_path = os.path.join(label_path, img_filename)
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize image
                    images.append(img)
                    labels.append(label_folder)  # Assuming each subfolder name is a label

    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
     # Normalize images to [0, 1] range
    images = images / 255.0
    images = images.reshape(-1, 224, 224, 1)  # Reshape for a CNN input

    # Map string labels to integers
    unique_labels = sorted(set(labels))  # Get unique labels and sort for consistent mapping
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_int[label] for label in labels])  # Convert labels to integers

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)   
    return X_train, X_test, y_train, y_test

# Test if the code works
if __name__ == "__main__":
    # Test loading images from folder
    images, labels = load_images_from_folder()
    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    
    # Test preprocessing data
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")


