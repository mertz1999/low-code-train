import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import datasets, models
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from lc_train.classification import train

# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.layers(x)

# Setup data loaders
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.ImageFolder('./data/train', transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

testset = datasets.ImageFolder('./data/train', transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False)

# Setup the model, loss criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train.fit(trainloader, testloader, model, optimizer, criterion, 12, resume=True, project='./')


loaded = torch.load('last.pth')
epoch_train_losses,epoch_train_accuracies,epoch_test_losses,epoch_test_accuracies = loaded['history']

plt.plot(epoch_train_losses)
plt.show()
