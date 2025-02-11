import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time

# Define Model
class XRaySorterModel(nn.Module):
    def __init__(self):
        super(XRaySorterModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)  # Reduced filters
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)  # Reduced filters
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # Adjusted for smaller input
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 classes
        self.dropout = nn.Dropout(0.3)  # Lower dropout to retain information

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)  # Adjusted for smaller input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Use CPU only
device = torch.device("cpu")

# Initialize model
model = XRaySorterModel().to(device)

# Define training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, dtype=torch.float16), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), "./aimodel.pth")
    print("Model saved successfully!")
    total_training_time = time.time() - start_time
    print(f"Total Training Time: {total_training_time / 3600:.2f} hours")

# Define evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, dtype=torch.float16), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

# Main execution
if __name__ == '__main__':
    train_dir = './DSETS/train'
    test_dir = './DSETS/test'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Reduced image size
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced batch size
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)  # Switched to SGD

    train_model(model, train_loader, criterion, optimizer, num_epochs=5)  # Reduced epochs
    evaluate_model(model, test_loader)
