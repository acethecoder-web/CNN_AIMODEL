import tkinter as tk
from tkinter import filedialog, Label, messagebox, Frame, Entry
from tkinter import ttk  
from PIL import Image, ImageTk
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime

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

# Initialize SQLite database
conn = sqlite3.connect("xray_results.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        patient_name TEXT,
        age INTEGER,
        prediction TEXT,
        timestamp TEXT
    )
""")
conn.commit()

# Function to save result in database
def save_to_database():
    global prediction_result, file_path
    name = name_entry.get()
    age = age_entry.get()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not name or not age:
        messagebox.showerror("Error", "Please fill in all fields!")
        return
    
    cursor.execute("INSERT INTO classifications (file_path, patient_name, age, prediction, timestamp) VALUES (?, ?, ?, ?, ?)",
                   (file_path, name, age, prediction_result, timestamp))
    conn.commit()
    messagebox.showinfo("Success", "Classification and patient data saved!")

def classify_image(selected_file):
    global file_path, prediction_result
    try:
        image = Image.open(selected_file)
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label = predicted.item()
        
        label_map = {0: "Normal", 1: "Pneumonia"}
        prediction_result = label_map[label]
        result_label.config(text=f"Predicted Class: {prediction_result}", fg="white", bg="#212121")
        
        save_button.pack(pady=10, padx=20, fill="x")
        
        # Show patient details form
        patient_form.pack(pady=10)
        prediction_entry.delete(0, tk.END)
        prediction_entry.insert(0, prediction_result)
        timestamp_label.config(text=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        messagebox.showerror("Error", "Invalid image file!")

def open_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((300, 300))
            img = ImageTk.PhotoImage(img)
            img_label.config(image=img)
            img_label.image = img
            classify_image(file_path)
        except:
            messagebox.showerror("Error", "Unable to open image!")

# GUI Setup
root = tk.Tk()
root.title("X-Ray Sorter App")
root.geometry("1000x400")  # Wider for form space
root.configure(bg="#212121")  

# Layout Frames
left_frame = Frame(root, bg="#212121", width=200)
left_frame.pack(side="left", fill="y")

middle_frame = Frame(root, bg="#303030", width=400)
middle_frame.pack(side="left", fill="both", expand=True)

right_frame = Frame(root, bg="#424242", width=400)
right_frame.pack(side="right", fill="both", expand=True)

# Buttons Section (Left Side)
title_label = Label(left_frame, text="X-Ray Sorter", font=("Arial", 16, "bold"), fg="white", bg="#212121")
title_label.pack(pady=20)

upload_button = ttk.Button(left_frame, text="Upload X-Ray Image", command=open_file)
upload_button.pack(pady=10, padx=20, fill="x")
upload_button.config(style="TButton")

save_button = ttk.Button(left_frame, text="Save to Database", command=save_to_database)
save_button.pack_forget()
save_button.config(style="TButton")

exit_button = ttk.Button(left_frame, text="Exit", command=root.quit)
exit_button.pack(pady=10, padx=20, fill="x")
exit_button.config(style="TButton")

# Image Preview Section
img_label = Label(middle_frame, bg="#303030", width=300, height=300)
img_label.pack(pady=20)
result_label = Label(middle_frame, text="Prediction Result", font=("Arial", 12), fg="white", bg="#303030")
result_label.pack(pady=10)

# Patient Details Form
patient_form = Frame(right_frame, bg="#424242")
Label(patient_form, text="Patient Details", font=("Arial", 14, "bold"), fg="white", bg="#424242").pack()
Label(patient_form, text="Full Name:", fg="white", bg="#424242").pack()
name_entry = Entry(patient_form)
name_entry.pack()
Label(patient_form, text="Age:", fg="white", bg="#424242").pack()
age_entry = Entry(patient_form)
age_entry.pack()
Label(patient_form, text="Classification:", fg="white", bg="#424242").pack()
prediction_entry = Entry(patient_form, state="readonly")
prediction_entry.pack()
timestamp_label = Label(patient_form, fg="white", bg="#424242")
timestamp_label.pack()

root.mainloop()
