import tkinter as tk
from tkinter import filedialog, Label, messagebox
from tkinter import ttk  # For themed widgets
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define the model
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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XRaySorterModel().to(device)
model.load_state_dict(torch.load("./aimodel.pth", map_location=device))  
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to classify the image
def classify_image(file_path):
    try:
        image = Image.open(file_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

        label_map = {0: "Normal", 1: "Pneumonia"}
        result_label.config(text=f"Predicted Class: {label_map[label]}", foreground="white", background="#212121")
    
    except Exception as e:
        messagebox.showerror("Error", "Invalid image file!")

# Function to open file
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            img = ImageTk.PhotoImage(img)
            img_label.config(image=img)
            img_label.image = img
            classify_image(file_path)
        except:
            messagebox.showerror("Error", "Unable to open image!")

# GUI Setup
root = tk.Tk()
root.title("X-Ray Sorter App")
root.geometry("400x500")
root.configure(bg="#212121")  # Dark theme

# Title Label
title_label = Label(root, text="X-Ray Sorter App", font=("Arial", 16, "bold"), fg="white", bg="#212121")
title_label.pack(pady=10)

# Image Label
img_label = Label(root, bg="#303030")
img_label.pack(pady=10)

# Result Label
result_label = Label(root, text="Prediction Result", font=("Arial", 12), fg="white", bg="#212121")
result_label.pack(pady=10)

# Upload Button with Styling
upload_button = ttk.Button(root, text="Upload X-Ray Image", command=open_file)
upload_button.pack(pady=10)

# Exit Button
exit_button = ttk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=10)

root.mainloop()
