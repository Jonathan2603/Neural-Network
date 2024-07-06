import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


# Set device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network architecture
class SimpleCNN(nn.Module):
      def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

      def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x
      

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Load and preprocess the Fashion MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize best accuracy
best_accuracy = 0.0

# Training loop
for epoch in range(epochs):
    model.train()
    total_train_correct = 0
    total_train_samples = 0
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()    
    
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train_samples += labels.size(0)
        total_train_correct += (predicted == labels).sum().item()

        total_loss += loss.item()

    train_accuracy = total_train_correct / total_train_samples
    average_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}")

    # Save the model if it has the best accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_accuracy': best_accuracy,
        }, 'simple_cnn_best_model.pth')


# Load the best model
best_model = SimpleCNN().to(device)
best_model_checkpoint = torch.load('simple_cnn_best_model.pth')
best_model.load_state_dict(best_model_checkpoint['model_state_dict'])
