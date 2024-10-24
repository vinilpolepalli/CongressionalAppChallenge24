import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose

def main():
    print("beginning a super cool program...")
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(in_channels=1024, out_channels=4096, kernel_size=3, stride=1, padding=1)
    #        self.conv4 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=1)
    #        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=4048, kernel_size=3, stride=1, padding=1)
    #        self.conv6 = nn.Conv2d(in_channels=4048, out_channels=4048*4, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Calculate the input size for the first fully connected layer
        # CIFAR-10 images are 32x32
            self.fc1 = nn.Linear(in_features=256*8*8, out_features=1024)
        #    self.fc2 = nn.Linear(in_features=4048, out_features=1024)
            self.fc2 = nn.Linear(in_features=1024, out_features=512)
        #    self.fc3 = nn.Linear(in_features=256, out_features=64)
            self.fc3 = nn.Linear(in_features=512, out_features=9)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = self.pool(self.relu(self.conv4(x)))
        #    x = self.pool(self.relu(self.conv5(x)))
        #    x = self.pool(self.relu(self.conv6(x)))
            x = x.view(-1, 256*8*8)  # Flatten the tensor
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            x = self.fc3(x)
        #    x = self.fc4(x)
        #    x = self.fc5(x)
            return x

# Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to desired size if needed
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dir = '/Users/henrycantor/Desktop/ML?/train'
    test_dir = '/Users/henrycantor/Desktop/ML?/test'
    test2_dir = '/Users/henrycantor/Desktop/ML?/val'
    batch_size = 1

# Load CIFAR-10 training and test datasets
    trainset = ImageFolder(root=train_dir, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = ImageFolder(root=test_dir, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    test2set = ImageFolder(root=test2_dir, transform=transform)
    test2loader = DataLoader(test2set, batch_size=batch_size, shuffle=False, num_workers=2)

    def train_model(model, criterion, optimizer, dataloader, num_epochs=7):
        print("training...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            model.train()  # Set model to training mode
            print("model training")
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Initialize the model, criterion, and optimizer
    model = SimpleCNN()
    print("initialized")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
    train_model(model, criterion, optimizer, trainloader)

    torch.save(model.state_dict(), 'simple_cnn.pth')

    def evaluate_model(model, dataloader):
        print("evaluating...")
        correct = 0
        total = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total:.2f}%')

# Evaluate the model on the test set
    evaluate_model(model, testloader)
    evaluate_model(model, test2loader)

# Save the trained model
    torch.save(model.state_dict(), 'crazy_cnn.pth')

if __name__ == '__main__':
    main()