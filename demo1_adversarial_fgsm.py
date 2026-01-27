# demo1_adversarial_fgsm.py
# Requirements: torch, torchvision, numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# Data
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# Train small model briefly
model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1):
    model.train()
    for X,y in trainloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

# Evaluate clean accuracy on test
model.eval()
correct=0
total=0
with torch.no_grad():
    for X,y in testloader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
print("Clean test accuracy:", correct/total)

# FGSM implementation
def fgsm_attack(model, X, y, eps=0.1):
    X_adv = X.clone().detach().to(device).requires_grad_(True)
    model.zero_grad()
    logits = model(X_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    grad = X_adv.grad.data
    X_adv = X_adv + eps * torch.sign(grad)
    X_adv = torch.clamp(X_adv, 0, 1)
    return X_adv.detach()

# Generate adversarial examples for the first batch of test data
dataiter = iter(testloader)
X,y = next(dataiter)
X,y = X.to(device), y.to(device)
X_adv = fgsm_attack(model, X, y, eps=0.2)

# Evaluate on adversarial examples
model.eval()
with torch.no_grad():
    adv_preds = model(X_adv).argmax(dim=1)
    adv_acc = (adv_preds == y).float().mean().item()
print("Adversarial batch accuracy (eps=0.2):", adv_acc)

# Show per-sample confidence drop for first 10 samples
softmax = nn.Softmax(dim=1)
with torch.no_grad():
    clean_conf = softmax(model(X))[:,0:10].max(dim=1)[0].cpu().numpy()
    adv_conf = softmax(model(X_adv))[:,0:10].max(dim=1)[0].cpu().numpy()
print("Sample confidence drops (first 10):")
for i in range(10):
    print(f"{i}: clean={clean_conf[i]:.3f} adv={adv_conf[i]:.3f} drop={clean_conf[i]-adv_conf[i]:.3f}")
