import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from aimodel import model, device

# Define the model (from aimodel.py)
class XRaySorterModel(nn.Module):
    def __init__(self):
        super(XRaySorterModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XRaySorterModel().to(device)
model.load_state_dict(torch.load("./aimodel.pth"))  # Replace with your model path
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
def classify_image(file_path):
    image = Image.open(file_path)
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the predicted class index
        label = predicted.item()

    # Map the class index to actual labels
    label_map = {0: "Normal", 1: "Pneumonia"}  # 0 -> Normal, 1 -> Pneumonia
    result_label.config(text=f"Predicted Class: {label_map[label]}")  # Display the label

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((150, 150))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
        classify_image(file_path)

# Set up GUI window
root = tk.Tk()
root.title("X-Ray Sorter App")

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="Prediction Result")
result_label.pack()

upload_button = tk.Button(root, text="Upload X-Ray Image", command=open_file)
upload_button.pack()

root.mainloop()
