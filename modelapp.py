import tkinter as tk
from tkinter import filedialog, Label, messagebox, Frame, Entry, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
import sqlite3
import torch
import torch.nn as nn
from torchvision import transforms, models
from datetime import datetime

# Color Variables
DARK_BG = "#212121"
DARK_FG = "white"
LIGHT_BG = "#f0f0f0"
LIGHT_FG = "black"
current_theme = "dark"  # Default theme

# Define the model
class XRaySorterModel(nn.Module):
    def __init__(self):
        super(XRaySorterModel, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)  # Modify first layer for grayscale
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # 4 Classes: Normal, Pneumonia, Tuberculosis, Not an X-ray
        )
    
    def forward(self, x):
        return self.model(x)

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
    if page == "dashboard":
        sorter_frame.pack_forget()
        dashboard_frame.pack(fill="both", expand=True)
        update_page_label("Dashboard")
        change_sidebar_buttons("dashboard")
        analytics_frame.pack(fill="both", expand=True)
    elif page == "sorter":
        dashboard_frame.pack_forget()
        sorter_frame.pack(fill="both", expand=True)
        update_page_label("Sorter")
        change_sidebar_buttons("sorter")
        analytics_frame.pack_forget()

# Function to update the page label
def update_page_label(page_name):
    page_label.config(text=page_name)

# Change sidebar buttons based on the page
def change_sidebar_buttons(page):
    for widget in sidebar.winfo_children():
        if isinstance(widget, ttk.Button):
            widget.destroy()
    
    if page == "dashboard":
        ttk.Button(sidebar, text="Dashboard", command=lambda: show_page("dashboard"), style="Bold.TButton").pack(pady=10, padx=20, fill="x")
        ttk.Button(sidebar, text="Sorter", command=lambda: show_page("sorter"), style="Bold.TButton").pack(pady=10, padx=20, fill="x")
    elif page == "sorter":
        ttk.Button(sidebar, text="Upload X-ray Image", command=open_file, style="Bold.TButton").pack(pady=10, padx=20, fill="x")
        ttk.Button(sidebar, text="Dashboard", command=lambda: show_page("dashboard"), style="Bold.TButton").pack(pady=10, padx=20, fill="x")
        ttk.Button(sidebar, text="Toggle Theme", command=toggle_theme, style="Bold.TButton").pack(pady=10, padx=20, fill="x")
        ttk.Button(sidebar, text="Exit", command=root.quit, style="Bold.TButton").pack(pady=10, padx=20, fill="x")

# Function to toggle themes
def toggle_theme():
    global current_theme
    if current_theme == "dark":
        current_theme = "light"
        root.configure(bg=LIGHT_BG)
        sidebar.configure(bg=LIGHT_BG)
        page_label.configure(bg=LIGHT_BG, fg=LIGHT_FG)
        for widget in root.winfo_children():
            widget.configure(bg=LIGHT_BG, fg=LIGHT_FG)
    else:
        current_theme = "dark"
        root.configure(bg=DARK_BG)
        sidebar.configure(bg=DARK_BG)
        page_label.configure(bg=DARK_BG, fg=DARK_FG)
        for widget in root.winfo_children():
            widget.configure(bg=DARK_BG, fg=DARK_FG)

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
            classify_image(file_path)
        except:
            messagebox.showerror("Error", "Unable to open image!")

# Function to classify image
def classify_image(file_path):
    image = Image.open(file_path).convert("L")  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    categories = ["Normal", "Pneumonia", "Tuberculosis", "Not an X-ray"]
    predicted_class = categories[predicted.item()]
    result_label.config(text=f"Prediction: {predicted_class}")
    disease_var.set(predicted_class)
    update_dashboard()
    fade_in(result_label)

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
    test_time = datetime.now()

    if patient_name and age and disease:
        save_to_db(patient_name, age, disease, test_time)
    else:
        messagebox.showwarning("Warning", "Please fill all fields!")

# Function to clear all input fields and image
def clear_data():
    img_label.config(image=None)
    img_label.image = None
    patient_name_var.set("")
    age_var.set("")
    disease_var.set("")
    result_label.config(text="Prediction Result")

# Function to fade in the result label
def fade_in(widget, step=1):
    current_color = widget.cget("fg")
    r, g, b = widget.winfo_rgb(current_color)
    new_color = f'#{r//256:02x}{g//256:02x}{b//256:02x}'
    widget.configure(fg=new_color)
    if step < 255:
        widget.after(10, fade_in, widget, step + 1)

# Main GUI Setup
root = tk.Tk()
root.title("X-Ray Sorter App")
root.geometry("800x400")
root.configure(bg=DARK_BG)

# Sidebar
sidebar = Frame(root, bg=DARK_BG, width=200)
sidebar.pack(side="left", fill="y")

title_label = Label(sidebar, text="X-Ray Sorter", font=("Arial", 16, "bold"), fg=DARK_FG, bg=DARK_BG)
title_label.pack(pady=20)

# Set the style for bold text
style = ttk.Style()
style.configure("Bold.TButton", font=("Arial", 12, "bold"))

# Page Label (this will show which page is active)
page_label = Label(root, text="Dashboard", font=("Arial", 14, "bold"), fg=DARK_FG, bg=DARK_BG)
page_label.pack(pady=20)

# Container for main content (Dashboard or Sorter)
dashboard_container = Frame(root, bg=DARK_BG, width=600, height=400)
dashboard_container.pack(fill="both", expand=True)

# Define the frames globally
dashboard_frame = Frame(dashboard_container, bg=DARK_BG)
sorter_frame = Frame(dashboard_container, bg=DARK_BG)

# Dashboard UI
dashboard_frame.pack(pady=20)

# Analytics Section (Only on Dashboard)
analytics_frame = Frame(dashboard_container, bg=DARK_BG)
analytics_frame.pack_forget()

# Statistics Section
stats_frame = Frame(analytics_frame, bg=DARK_BG)
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
    not_xray_count = cursor.fetchone()[0]
    return total_tests, pneumonia_count, tuberculosis_count, not_xray_count

def update_dashboard():
    total_tests, pneumonia_count, tuberculosis_count, not_xray_count = get_statistics()
    
    for widget in stats_frame.winfo_children():
        widget.destroy()
    
    stats = [
        ("Total Tests Conducted", total_tests, "#4CAF50"),
        ("Pneumonia Cases", pneumonia_count, "#F44336"),
        ("Tuberculosis Cases", tuberculosis_count, "#FF9800"),
        ("Not an X-ray Cases", not_xray_count, "#2196F3")
    ]
    
    for stat in stats:
        stat_frame = Frame(stats_frame, bg=stat[2], width=180, height=100, padx=10, pady=10)
        stat_frame.pack(side="top", padx=10, pady=10, anchor="center")
        
        Label(stat_frame, text=stat[0], font=("Arial", 12), fg="white", bg=stat[2]).pack()
        Label(stat_frame, text=stat[1], font=("Arial", 16, "bold"), fg="white", bg=stat[2]).pack()

# Update the dashboard to show the analytics
update_dashboard()

# Sorter Section (X-ray image preview and result)
img_label = Label(sorter_frame, bg=DARK_BG, width=300, height=300)
img_label.pack(pady=(20, 10))

result_label = Label(sorter_frame, text="Prediction Result", font=("Arial", 12), fg=DARK_FG, bg=DARK_BG)
result_label.pack(pady=10)

# Create input fields for patient details
patient_name_var = StringVar()
age_var = StringVar()
disease_var = StringVar()

# Patient Name
Label(sorter_frame, text="Patient Name:", bg=DARK_BG, fg=DARK_FG).pack(pady=5)
Entry(sorter_frame, textvariable=patient_name_var).pack(pady=5)

# Age
Label(sorter_frame, text="Age:", bg=DARK_BG, fg=DARK_FG).pack(pady=5)
Entry(sorter_frame, textvariable=age_var).pack(pady=5)

# Disease (Auto-filled)
Label(sorter_frame, text="Disease:", bg=DARK_BG, fg=DARK_FG).pack(pady=5)
Entry(sorter_frame, textvariable=disease_var, state='readonly').pack(pady=5)

# Save Button
ttk.Button(sorter_frame, text="Save Test Details", command=save_details, style="Bold.TButton").pack(pady=10)

# Clear Data Button
ttk.Button(sorter_frame, text="Clear Data", command=clear_data, style="Bold.TButton").pack(pady=10)

# Initially show the dashboard
show_page("dashboard")

# Run the application
root.mainloop()
