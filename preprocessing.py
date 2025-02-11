import os
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split

# Image size to match CNN input
IMG_SIZE = (224, 224)

# Dataset folder path
folder_path = "./DSETS/train"

# Batch size for processing
BATCH_SIZE = 100  

def load_images_from_folder(img_size=IMG_SIZE, batch_size=BATCH_SIZE):  
    images = []
    labels = []
    total_images = sum(len(files) for _, _, files in os.walk(folder_path))
    processed_count = 0
    batch_number = 1
    start_time = time.time()

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        
        if os.path.isdir(label_path):
            for img_filename in os.listdir(label_path):
                img_path = os.path.join(label_path, img_filename)
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize to 224x224
                    images.append(img)
                    labels.append(label_folder)

                processed_count += 1
                
                # Process in batches
                if len(images) >= batch_size:
                    print(f"Processing batch {batch_number}: {processed_count}/{total_images} images completed")
                    yield np.array(images), np.array(labels)  # Yield batch
                    images, labels = [], []  # Reset batch
                    batch_number += 1

    # Process any remaining images
    if images:
        print(f"Processing final batch {batch_number}: {processed_count}/{total_images} images completed")
        yield np.array(images), np.array(labels)

    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")

def preprocess_data(images, labels):
    images = images / 255.0  # Normalize images
    images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)  # Reshape for CNN input

    # Convert labels to integer encoding
    unique_labels = sorted(set(labels))  # Ensure consistent label ordering
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_int[label] for label in labels])

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)   
    
    return X_train, X_test, y_train, y_test, label_to_int  

# Run preprocessing in batches
if __name__ == "__main__":
    total_batches = 0
    for images, labels in load_images_from_folder():
        X_train, X_test, y_train, y_test, label_map = preprocess_data(images, labels)
        total_batches += 1
        print(f"Batch {total_batches} processed. Training data: {X_train.shape}, Test data: {X_test.shape}")
