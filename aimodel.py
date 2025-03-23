import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time

# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNetB0 (Pretrained)
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

# Initialize Model
model = XRaySorterModel().to(device)

# Optimizer & Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005)
scaler = torch.cuda.amp.GradScaler()

# Training Function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, accumulation_steps=2):
    start_time = time.time()
    best_loss = float("inf")
    patience, counter = 3, 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
        
        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), "efficientnet_xray.pth")
            print("Model saved!")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

        print(f"Epoch Time: {time.time() - epoch_start:.2f} sec")
    
    print(f"Total Training Time: {(time.time() - start_time) / 3600:.2f} hours")

# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load Datasets
train_dataset = datasets.ImageFolder(root='./DSETS/train', transform=transform)
test_dataset = datasets.ImageFolder(root='./DSETS/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Train & Evaluate
if __name__ == '__main__':
    train_model(model, train_loader, criterion, optimizer, num_epochs=3)
    evaluate_model(model, test_loader)
