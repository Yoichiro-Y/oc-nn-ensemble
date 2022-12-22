import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for the data
transform = transforms.Compose([transforms.ToTensor()])

# Load the training data from MNIST
train_dataset = MNIST(root='.', train=True, transform=transform, download=True)

# Create a DataLoader for the training data
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the one-class neural network
class OCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize the model and move it to the device
model = OCNN(28*28, 128, 1).to(device)

# Define the loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(10):
    for i, (images, labels) in enumerate(train_dataloader):
        # Move the data to the device
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = loss_fn(outputs, labels.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/10], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'ocnn.pth')
