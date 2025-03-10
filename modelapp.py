import tkinter as tk
from tkinter import filedialog, Label, messagebox, Frame, Entry, StringVar
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
        self.fc3 = nn.Linear(128, 4)  # Change from 3 to 4 for the new category
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
model.load_state_dict(torch.load("./efficientnet_xray.pth", map_location=device))  
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
        prediction TEXT,
        patient_name TEXT,
        age INTEGER,
        disease TEXT,
        test_time DATETIME
    )
""")
conn.commit()

# Function to switch pages and update the label

def show_page(page):
    # Hide all frames before showing the selected page
    if page == "dashboard":
        sorter_frame.pack_forget()
        dashboard_frame.pack(fill="both", expand=True)
        update_page_label("Dashboard")  # Update the page label
        change_sidebar_buttons("dashboard")
        analytics_frame.pack(fill="both", expand=True)  # Show analytics section
    elif page == "sorter":
        dashboard_frame.pack_forget()
        sorter_frame.pack(fill="both", expand=True)
        update_page_label("Sorter")  # Update the page label
        change_sidebar_buttons("sorter")
        analytics_frame.pack_forget()  # Hide analytics section when on sorter page

# Function to update the page label
def update_page_label(page_name):
    page_label.config(text=page_name)

# Change sidebar buttons based on the page
def change_sidebar_buttons(page):
    # Clear existing sidebar buttons
    for widget in sidebar.winfo_children():
        if isinstance(widget, ttk.Button):
            widget.destroy()
    
    # Add buttons based on the selected page
    if page == "dashboard":
        ttk.Button(sidebar, text="Dashboard", command=lambda: show_page("dashboard"), style="Bold.TButton").pack(pady=10, padx=20, fill="x")
        ttk.Button(sidebar, text="Sorter", command=lambda: show_page("sorter"), style="Bold.TButton").pack(pady=10, padx=20, fill="x")
    elif page == "sorter":
        ttk.Button(sidebar, text="Upload X-ray Image", command=open_file, style="Bold.TButton").pack(pady=10, padx=20, fill="x")
        ttk.Button(sidebar, text="Dashboard", command=lambda: show_page("dashboard"), style="Bold.TButton").pack(pady=10, padx=20, fill="x")
        ttk.Button(sidebar, text="Exit", command=root.quit, style="Bold.TButton").pack(pady=10, padx=20, fill="x")

# Function to open and display X-ray image and predict
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
            result_label.config(text="Processing...")
            classify_image(file_path)  # Classify image after displaying
        except:
            messagebox.showerror("Error", "Unable to open image!")

# Function to classify image
def classify_image(file_path):
    image = Image.open(file_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Class prediction logic (0: 'Normal', 1: 'Pneumonia', 2: 'Tuberculosis', 3: 'Not an X-ray')
    categories = ["Normal", "Pneumonia", "Tuberculosis", "Not an X-ray"]
    predicted_class = categories[predicted.item()]
    result_label.config(text=f"Prediction: {predicted_class}")
    disease_var.set(predicted_class)  # Automatically set the disease field
    update_dashboard()  # Update the dashboard after classifying

# Save result to database
def save_to_db(patient_name, age, disease, test_time):
    cursor.execute("INSERT INTO classifications (file_path, prediction, patient_name, age, disease, test_time) VALUES (?, ?, ?, ?, ?, ?)", 
                   (file_path, disease, patient_name, age, disease, test_time))
    conn.commit()
    messagebox.showinfo("Success", "Test details saved successfully!")

# Function to handle saving the test details
def save_details():
    patient_name = patient_name_var.get()
    age = age_var.get()
    disease = disease_var.get()
    test_time = datetime.now()  # Get current date and time

    if patient_name and age and disease:
        save_to_db(patient_name, age, disease, test_time)
    else:
        messagebox.showwarning("Warning", "Please fill all fields!")

# Function to clear all input fields and image
def clear_data():
    img_label.config(image=None)  # Clear the image
    img_label.image = None  # Remove reference to the image
    patient_name_var.set("")  # Clear patient name
    age_var.set("")  # Clear age
    disease_var.set("")  # Clear disease
    result_label.config(text="Prediction Result")  # Clear the result label

# Main GUI Setup
root = tk.Tk()
root.title("X-Ray Sorter App")
root.geometry("800x400")
root.configure(bg="#212121")

# Sidebar
sidebar = Frame(root, bg="#212121", width=200)
sidebar.pack(side="left", fill="y")

title_label = Label(sidebar, text="X-Ray Sorter", font=("Arial", 16, "bold"), fg="white", bg="#212121")
title_label.pack(pady=20)

# Set the style for bold text
style = ttk.Style()
style.configure("Bold.TButton", font=("Arial", 12, "bold"))

# Page Label (this will show which page is active)
page_label = Label(root, text="Dashboard", font=("Arial", 14, "bold"), fg="white", bg="#212121")
page_label.pack(pady=20)

# Container for main content (Dashboard or Sorter)
dashboard_container = Frame(root, bg="#303030", width=600, height=400)
dashboard_container.pack(fill="both", expand=True)

# Define the frames globally
dashboard_frame = Frame(dashboard_container, bg="#303030")
sorter_frame = Frame(dashboard_container, bg="#303030")

# Dashboard UI
dashboard_frame.pack(pady=20)

# Analytics Section (Only on Dashboard)
analytics_frame = Frame(dashboard_container, bg="#303030")
analytics_frame.pack_forget()  # Initially hide the analytics section

# Statistics Section
stats_frame = Frame(analytics_frame, bg="#303030")
stats_frame.pack(pady=10, fill="both", expand=True)

# Function to fetch statistics
def get_statistics():
    cursor.execute("SELECT COUNT(*) FROM classifications")
    total_tests = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM classifications WHERE prediction = 'Pneumonia'")
    pneumonia_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM classifications WHERE prediction = 'Tuberculosis'")
    tuberculosis_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM classifications WHERE prediction = 'Not an X-ray'")
    not_xray_count = cursor.fetchone()[0]  # New count for 'Not an X-ray'
    return total_tests, pneumonia_count, tuberculosis_count, not_xray_count

def update_dashboard():
    total_tests, pneumonia_count, tuberculosis_count, not_xray_count = get_statistics()
    
    # Clear existing statistics labels before updating
    for widget in stats_frame.winfo_children():
        widget.destroy()
    
    stats = [
        ("Total Tests Conducted", total_tests),
        ("Pneumonia Cases", pneumonia_count),
        ("Tuberculosis Cases", tuberculosis_count),
        ("Not an X-ray Cases", not_xray_count)  # New statistic
    ]
    
    # Center the statistics section
    for stat in stats:
        stat_frame = Frame(stats_frame, bg="#424242", width=180, height=100, padx=10, pady=10)
        stat_frame.pack(side="top", padx=10, pady=10, anchor="center")  # Center the stat frame
        
        Label(stat_frame, text=stat[0], font=("Arial", 12), fg="white", bg="#424242").pack()
        Label(stat_frame, text=stat[1], font=("Arial", 16, "bold"), fg="white", bg="#424242").pack()

# Update the dashboard to show the analytics
update_dashboard()

# Sorter Section (X-ray image preview and result)
img_label = Label(sorter_frame, bg="#303030", width=300, height=300)
img_label.pack(pady=(20, 10))  # Add padding to the top and bottom

result_label = Label(sorter_frame, text="Prediction Result", font=("Arial", 12), fg="white", bg="#303030")
result_label.pack(pady=10)

# Create input fields for patient details
patient_name_var = StringVar()
age_var = StringVar()
disease_var = StringVar()

# Patient Name
Label(sorter_frame, text="Patient Name:", bg="#303030", fg="white").pack(pady=5)
Entry(sorter_frame, textvariable=patient_name_var).pack(pady=5)

# Age
Label(sorter_frame, text="Age:", bg="#303030", fg="white").pack(pady=5)
Entry(sorter_frame, textvariable=age_var).pack(pady=5)

# Disease (Auto-filled)
Label(sorter_frame, text="Disease:", bg="#303030", fg="white").pack(pady=5)
Entry(sorter_frame, textvariable=disease_var, state='readonly').pack(pady=5)  # Make it read-only

# Save Button
ttk.Button(sorter_frame, text="Save Test Details", command=save_details, style="Bold.TButton").pack(pady=10)

# Clear Data Button
ttk.Button(sorter_frame, text="Clear Data", command=clear_data, style="Bold.TButton").pack(pady=10)

# Initially show the dashboard
show_page("dashboard")

# Run the application
root.mainloop()
