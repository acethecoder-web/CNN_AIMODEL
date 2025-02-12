import tkinter as tk
from tkinter import filedialog, Label, messagebox, Frame, Entry
from tkinter import ttk  
from PIL import Image, ImageTk
import mysql.connector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime

# MySQL Connection Setup
def connect_db():
    return mysql.connector.connect(
        host="localhost", 
        user="root", 
        password="",  # Update with your MySQL password if needed
        database="xraysorter"
    )

def initialize_database():
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS disease (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50) UNIQUE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            age INT,
            disease VARCHAR(50),
            date_time DATETIME,
            FOREIGN KEY (disease) REFERENCES disease(name)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary (
            id INT AUTO_INCREMENT PRIMARY KEY,
            total_tests INT DEFAULT 0,
            total_pneumonia INT DEFAULT 0,
            total_tuberculosis INT DEFAULT 0,
            total_normal INT DEFAULT 0
        )
    """)
    
    # Insert disease categories if not exists
    for disease in ["Normal", "Pneumonia", "Tuberculosis"]:
        cursor.execute("INSERT IGNORE INTO disease (name) VALUES (%s)", (disease,))
    
    # Ensure summary table has one row
    cursor.execute("SELECT COUNT(*) FROM summary")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO summary (total_tests, total_pneumonia, total_tuberculosis, total_normal) VALUES (0, 0, 0, 0)")
    
    conn.commit()
    cursor.close()
    conn.close()

initialize_database()

# Define CNN Model
class XRaySorterModel(nn.Module):
    def __init__(self):
        super(XRaySorterModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 classes
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

# Function to save result in database
def save_to_database():
    global prediction_result
    name = name_entry.get()
    age = age_entry.get()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not name or not age:
        messagebox.showerror("Error", "Please fill in all fields!")
        return
    
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO patient_data (name, age, disease, date_time) VALUES (%s, %s, %s, %s)",
                   (name, age, prediction_result, timestamp))
    cursor.execute("UPDATE summary SET total_tests = total_tests + 1, "
                   f"total_{prediction_result.lower()} = total_{prediction_result.lower()} + 1")
    conn.commit()
    cursor.close()
    conn.close()
    messagebox.showinfo("Success", "Classification and patient data saved!")

def classify_image(selected_file):
    global prediction_result
    try:
        image = Image.open(selected_file)
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label = predicted.item()
        
        label_map = {0: "Normal", 1: "Pneumonia", 2: "Tuberculosis"}
        prediction_result = label_map[label]
        result_label.config(text=f"Predicted Class: {prediction_result}", fg="white", bg="#212121")
        save_button.pack(pady=10, padx=20, fill="x")
        patient_form.pack(pady=10)
        prediction_entry.delete(0, tk.END)
        prediction_entry.insert(0, prediction_result)
        timestamp_label.config(text=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        messagebox.showerror("Error", "Invalid image file!")

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        try:
            img = Image.open(file_path).resize((300, 300))
            img = ImageTk.PhotoImage(img)
            img_label.config(image=img)
            img_label.image = img
            classify_image(file_path)
        except:
            messagebox.showerror("Error", "Unable to open image!")

# GUI Setup
root = tk.Tk()
root.title("X-Ray Sorter App")
root.geometry("1000x400")  
root.configure(bg="#212121")  

left_frame = Frame(root, bg="#212121", width=200)
left_frame.pack(side="left", fill="y")

upload_button = ttk.Button(left_frame, text="Upload X-Ray Image", command=open_file)
upload_button.pack(pady=10, padx=20, fill="x")

save_button = ttk.Button(left_frame, text="Save to Database", command=save_to_database)
save_button.pack_forget()

root.mainloop()
