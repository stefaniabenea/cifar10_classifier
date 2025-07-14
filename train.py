import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from model import SimpleCNN
from utils import evaluate

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download= True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train set size: {len(train_loader.dataset)}")
print(f"Test set size: {len(test_loader.dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

num_epochs=10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss/len(train_loader)    
    
    
    # evaluate
    test_acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}, test accuracy: {test_acc:.2f}%")

 
