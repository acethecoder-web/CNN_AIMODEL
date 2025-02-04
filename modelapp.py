import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define the model (from aimodel.py)
class XRaySorterModel(nn.Module):
    def __init__(self):
        super(XRaySorterModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adjust fc1 input size for 224x224 images
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjusted from 128*28*28
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: 112x112
        x = self.pool(F.relu(self.conv2(x)))  # Output: 56x56
        x = self.pool(F.relu(self.conv3(x)))  # Output: 28x28
        x = x.view(-1, 128 * 28 * 28)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XRaySorterModel().to(device)
model.load_state_dict(torch.load("./aimodel.pth", map_location=device))  
model.eval()

# Define image transformations (224x224 instead of 28x28)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Updated to match preprocessing
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def classify_image(file_path):
    image = Image.open(file_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = predicted.item()

    label_map = {0: "Normal", 1: "Pneumonia"}  
    result_label.config(text=f"Predicted Class: {label_map[label]}")  

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((150, 150))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
        classify_image(file_path)

# Set up GUI
root = tk.Tk()
root.title("X-Ray Sorter App")

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="Prediction Result")
result_label.pack()

upload_button = tk.Button(root, text="Upload X-Ray Image", command=open_file)
upload_button.pack()

root.mainloop()
