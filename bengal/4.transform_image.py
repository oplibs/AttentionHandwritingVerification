#!/usr/bin/python
# -*- coding: utf-8 -*
##############################################################
# File Name: 4.transform_image.py
# Author:
# mail:
# Created Time: Sun Mar  2 15:33:42 2025
# Brief:
##############################################################

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchvision import transforms

# Training transformations
train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure images are single-channel
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

# Validation/Test transformations
val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])
# Load datasets
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='data/val', transform=val_transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=val_transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data loaders created successfully!")

data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"Batch size: {images.size(0)}, Image shape: {images[0].shape}")


import torch
import torch.nn as nn
import timm

# Load the pre-trained ViT model
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Modify the input layer to accept single-channel images
model.patch_embed.proj = nn.Conv2d(
    in_channels=1,
    out_channels=model.patch_embed.proj.out_channels,
    kernel_size=model.patch_embed.proj.kernel_size,
    stride=model.patch_embed.proj.stride,
    padding=model.patch_embed.proj.padding,
    bias=model.patch_embed.proj.bias is not None
)

# Adjust the classification head for binary classification
num_features = model.head.in_features
model.head = nn.Linear(num_features, 2)  # Two output classes

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Model setup completed successfully!")
device
print("Modified input layer:")
print(model.patch_embed.proj)


def train_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    iterations = 0
    print("Start train_one_epoch");

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iterations = iterations + 1
        if (iterations % 10) == 0:
            print("{} items done!".format(iterations))

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, criterion, dataloader, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"Image batch shape: {images.shape}")
# Expected output: [batch_size, 1, 224, 224]

import time
import os

num_epochs = 20  # You can adjust this number

def train_model(model, weight_file, best_val_acc):
    for epoch in range(num_epochs):
        print("Current accuracy: {}".format(best_val_acc));
        start_time = time.time()

        # Training phase
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the best model weights
            torch.save(model.state_dict(), model_weights_file)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Epoch Time: {epoch_time:.2f} seconds\n')
    print('Training complete.')
    return best_val_acc;


best_val_acc = 0.0
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
model_weights_file='best_model.pth'

if os.path.exists(model_weights_file):
    print("Model file exists, loading ......")
    model.load_state_dict(torch.load(model_weights_file))
else:
    print("Model file does not exists, will re train ......")
    best_val_acc=train_model(model, model_weights_file, best_val_acc);
    print("Accuracy after trainning: {}".format(best_val_acc));

test_loss, test_acc = validate(model, criterion, test_loader, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
