import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from PIL import Image
import time
# Define Model
class XRaySorterModel(nn.Module):
    def __init__(self):
        super(XRaySorterModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
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


# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move it to the device
model = XRaySorterModel().to(device)
model.load_state_dict(torch.load("./aimodel.py"))
model.eval()

# Save the trained model's weights to a .pth file
torch.save(model.state_dict(),"./")

# Define training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    start_time = time.time()  # Record start time
    for epoch in range(num_epochs):
        epoch_start = time.time()  # Record epoch start time
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:  # Prints every 10 batches

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/len(train_loader)}")

  # Calculate time taken for this epoch
        epoch_time = time.time() - epoch_start
        remaining_time = epoch_time * (num_epochs - (epoch + 1))  # Estimate remaining time

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/len(train_loader)}")
        print(f"Epoch Time: {epoch_time:.2f} seconds | Estimated Time Remaining: {remaining_time / 3600:.2f} hours")

    total_training_time = time.time() - start_time  # Calculate total training time
    print(f"Total Training Time: {total_training_time / 3600:.2f} hours")

# Define evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

# Main execution
if __name__ == '__main__':
    #Define dataset directories (Replace these with your actual paths)
    train_dir = './DSETS/train'  # Replace with your train dataset directory
    test_dir = './DSETS/test'    # Replace with your test dataset directory

    # Define transformations for data preprocessing
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.Grayscale(num_output_channels=1),  # Ensure images are in grayscale (if they aren't)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the images (adjust mean and std for your data)
    ])

   #  Load the dataset (assuming dataset is structured into 'train' and 'test' folders)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

     #Initialize model, criterion, optimizer
    model = XRaySorterModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    evaluate_model(model, test_loader)

