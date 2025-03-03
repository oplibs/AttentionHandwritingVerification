#remove folder
#%rm -rf sample_data/
# %rm -rf BHSig260-Bengali/
# %rm -rf BHSig260-Hindi/
# %rm -rf datasets/
# !pip install kagglehub
# import kagglehub

# # Download the dataset to the Colab directory
# path = kagglehub.dataset_download("ishanikathuria/handwritten-signature-datasets", path="./datasets")

# print("Path to dataset files:", path)

#  import kagglehub

#  # Download latest version
#  path = kagglehub.dataset_download("ishanikathuria/handwritten-signature-datasets")

#  print("Path to dataset files:", path)
#  ! pwd
#  ! mv /root/.cache/kagglehub/datasets/ishanikathuria/handwritten-signature-datasets/versions/3 /content/

#  ! pwd
#  ! mv /content/3/BHSig260-Bengali/BHSig260-Bengali/ /content
#  ! mv /content/3/BHSig260-Hindi/BHSig260-Hindi /content/
import os

# Define the base directory
base_dir = "data"

# Define the splits and classes
splits = ["train", "val", "test"]
classes = ["genuine", "forged"]

# Create the directory structure
for split in splits:
    for cls in classes:
        folder = os.path.join(base_dir, split, cls)
        os.makedirs(folder, exist_ok=True)

print("Directory structure created!")

#  ! pwd
#  import os
#  import shutil
#  import random

#  # Define the source directory for the Bengali dataset
#  source_dir = "BHSig260-Bengali"

#  # Define the ratios for splitting
#  train_ratio = 0.7
#  val_ratio = 0.15
#  test_ratio = 0.15

#  # Define the base directory for the split data
#  base_dir = "data"

#  # Define the splits and classes
#  splits = ["train", "val", "test"]
#  classes = ["genuine", "forged"]

#  # Create the directory structure if it doesn't exist
#  for split in splits:
    #  for cls in classes:
        #  folder = os.path.join(base_dir, split, cls)
        #  os.makedirs(folder, exist_ok=True)

#  # Loop through each person's directory in the source directory
#  for person_dir in os.listdir(source_dir):
    #  person_path = os.path.join(source_dir, person_dir)

    #  # Get a list of all signature files for this person
    #  signature_files = os.listdir(person_path)
    #  num_signatures = len(signature_files)

    #  # Calculate the number of signatures for each split
    #  num_train = int(train_ratio * num_signatures)
    #  num_val = int(val_ratio * num_signatures)
    #  num_test = num_signatures - num_train - num_val

    #  # Shuffle the signature files randomly
    #  random.shuffle(signature_files)

    #  # Split the signature files into train, val, and test sets
    #  train_files = signature_files[:num_train]
    #  val_files = signature_files[num_train:num_train + num_val]
    #  test_files = signature_files[num_train + num_val:]

    #  # Copy the files to the corresponding folders
    #  for file in train_files:
        #  src_path = os.path.join(person_path, file)
        #  dst_path = os.path.join(base_dir, "train", "genuine" if "G" in file else "forged", file)
        #  shutil.copy(src_path, dst_path)

    #  for file in val_files:
        #  src_path = os.path.join(person_path, file)
        #  dst_path = os.path.join(base_dir, "val", "genuine" if "G" in file else "forged", file)
        #  shutil.copy(src_path, dst_path)

    #  for file in test_files:
        #  src_path = os.path.join(person_path, file)
        #  dst_path = os.path.join(base_dir, "test", "genuine" if "G" in file else "forged", file)
        #  shutil.copy(src_path, dst_path)

#  print("Dataset split and copied successfully!")

#  import os

#  # Define the base directory
#  base_dir = "data"

#  # Loop through the splits and classes
#  for split in ["train", "val", "test"]:
    #  for cls in ["genuine", "forged"]:
        #  folder_path = os.path.join(base_dir, split, cls)
        #  num_files = len(os.listdir(folder_path))
        #  print(f"Number of files in {split}/{cls}: {num_files}")



#  !pip install torch torchvision

#  from torchvision import transforms
#  from PIL import Image
#  import os

#  # Define resizing transformation
#  resize_transform = transforms.Compose([
    #  transforms.Resize((224, 224))
#  ])

#  # Loop through dataset folders
#  base_dir = "data"
#  for split in ["train", "val", "test"]:
    #  for cls in ["genuine", "forged"]:
        #  folder = os.path.join(base_dir, split, cls)
        #  for file in os.listdir(folder):
            #  if file.endswith(".png"):
                #  img_path = os.path.join(folder, file)
                #  img = Image.open(img_path)
                #  img_resized = resize_transform(img)
                #  img_resized.save(img_path)  # Overwrite with resized image

#  print("Resizing completed!")
#  from torchvision import transforms
#  from PIL import Image
#  import os

#  # Define resizing transformation
#  resize_transform = transforms.Compose([
    #  transforms.Resize((224, 224))
#  ])

#  # Loop through dataset folders
#  base_dir = "data"
#  for split in ["train", "val", "test"]:
    #  for cls in ["genuine", "forged"]:
        #  folder = os.path.join(base_dir, split, cls)
        #  for file in os.listdir(folder):
            #  if file.endswith(".tif"):
                #  img_path = os.path.join(folder, file)
                #  img = Image.open(img_path)
                #  img_resized = resize_transform(img)
                #  img_resized.save(img_path)  # Overwrite with resized image

#  print("Resizing completed!")

#  from torchvision import datasets, transforms
#  from torch.utils.data import DataLoader

#  from torchvision import transforms

#  # Training transformations
#  train_transform = transforms.Compose([
    #  transforms.Grayscale(num_output_channels=1),  # Ensure images are single-channel
    #  transforms.Resize((224, 224)),
    #  transforms.RandomRotation(degrees=5),
    #  transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    #  transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.5], std=[0.5])
#  ])

#  # Validation/Test transformations
#  val_transform = transforms.Compose([
    #  transforms.Grayscale(num_output_channels=1),
    #  transforms.Resize((224, 224)),
    #  transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.5], std=[0.5])
#  ])
#  # Load datasets
#  train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
#  val_dataset = datasets.ImageFolder(root='data/val', transform=val_transform)
#  test_dataset = datasets.ImageFolder(root='data/test', transform=val_transform)

#  # Create DataLoaders
#  batch_size = 32
#  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#  print("Data loaders created successfully!")
#  from torchvision import datasets, transforms
#  from torch.utils.data import DataLoader

#  from torchvision import transforms

#  # Training transformations
#  train_transform = transforms.Compose([
    #  transforms.Grayscale(num_output_channels=1),  # Ensure images are single-channel
    #  transforms.Resize((224, 224)),
    #  transforms.RandomRotation(degrees=5),
    #  transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    #  transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.5], std=[0.5])
#  ])

#  # Validation/Test transformations
#  val_transform = transforms.Compose([
    #  transforms.Grayscale(num_output_channels=1),
    #  transforms.Resize((224, 224)),
    #  transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.5], std=[0.5])
#  ])
#  # Load datasets
#  train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
#  val_dataset = datasets.ImageFolder(root='data/val', transform=val_transform)
#  test_dataset = datasets.ImageFolder(root='data/test', transform=val_transform)

#  # Create DataLoaders
#  batch_size = 32
#  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#  print("Data loaders created successfully!")
#  # Verify a single batch
#  import matplotlib.pyplot as plt

#  # Get one batch of training data
#  data_iter = iter(train_loader)
#  images, labels = next(data_iter)

#  # Plot the first image in the batch
#  plt.imshow(images[0].permute(1, 2, 0).squeeze(), cmap="gray")  # Adjust for grayscale
#  plt.title(f"Label: {labels[0]}")
#  plt.axis("off")
#  plt.show()

#  print(f"Batch size: {images.size(0)}, Image shape: {images[0].shape}")
#  import torch
#  import torch.nn as nn
#  import timm

#  # Load the pre-trained ViT model
#  model = timm.create_model('vit_base_patch16_224', pretrained=True)

#  # Modify the input layer to accept single-channel images
#  model.patch_embed.proj = nn.Conv2d(
    #  in_channels=1,
    #  out_channels=model.patch_embed.proj.out_channels,
    #  kernel_size=model.patch_embed.proj.kernel_size,
    #  stride=model.patch_embed.proj.stride,
    #  padding=model.patch_embed.proj.padding,
    #  bias=model.patch_embed.proj.bias is not None
#  )

#  # Adjust the classification head for binary classification
#  num_features = model.head.in_features
#  model.head = nn.Linear(num_features, 2)  # Two output classes

#  # Move model to device
#  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#  model = model.to(device)

#  # Define loss function and optimizer
#  criterion = nn.CrossEntropyLoss()
#  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#  print("Model setup completed successfully!")
#  device
#  print("Modified input layer:")
#  print(model.patch_embed.proj)


#  def train_one_epoch(model, optimizer, criterion, dataloader, device):
    #  model.train()  # Set model to training mode
    #  running_loss = 0.0
    #  correct = 0
    #  total = 0

    #  for images, labels in dataloader:
        #  images = images.to(device)
        #  labels = labels.to(device)

        #  # Zero the parameter gradients
        #  optimizer.zero_grad()

        #  # Forward pass
        #  outputs = model(images)
        #  loss = criterion(outputs, labels)

        #  # Backward pass and optimize
        #  loss.backward()
        #  optimizer.step()

        #  # Statistics
        #  running_loss += loss.item() * images.size(0)
        #  _, predicted = torch.max(outputs.data, 1)
        #  total += labels.size(0)
        #  correct += (predicted == labels).sum().item()

    #  epoch_loss = running_loss / total
    #  epoch_acc = 100 * correct / total

    #  return epoch_loss, epoch_acc
#  def validate(model, criterion, dataloader, device):
    #  model.eval()  # Set model to evaluation mode
    #  running_loss = 0.0
    #  correct = 0
    #  total = 0

    #  with torch.no_grad():  # Disable gradient computation
        #  for images, labels in dataloader:
            #  images = images.to(device)
            #  labels = labels.to(device)

            #  # Forward pass
            #  outputs = model(images)
            #  loss = criterion(outputs, labels)

            #  # Statistics
            #  running_loss += loss.item() * images.size(0)
            #  _, predicted = torch.max(outputs.data, 1)
            #  total += labels.size(0)
            #  correct += (predicted == labels).sum().item()

    #  epoch_loss = running_loss / total
    #  epoch_acc = 100 * correct / total

    #  return epoch_loss, epoch_acc
#  # Get one batch of data
#  data_iter = iter(train_loader)
#  images, labels = next(data_iter)

#  print(f"Image batch shape: {images.shape}")
#  # Expected output: [batch_size, 1, 224, 224]
#  import time

#  num_epochs = 20  # You can adjust this number
#  best_val_acc = 0.0

#  train_losses = []
#  train_accuracies = []
#  val_losses = []
#  val_accuracies = []

#  for epoch in range(num_epochs):
    #  start_time = time.time()

    #  # Training phase
    #  train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
    #  train_losses.append(train_loss)
    #  train_accuracies.append(train_acc)

    #  # Validation phase
    #  val_loss, val_acc = validate(model, criterion, val_loader, device)
    #  val_losses.append(val_loss)
    #  val_accuracies.append(val_acc)

    #  # Check if this is the best model so far
    #  if val_acc > best_val_acc:
        #  best_val_acc = val_acc
        #  # Save the best model weights
        #  torch.save(model.state_dict(), 'best_model.pth')

    #  end_time = time.time()
    #  epoch_time = end_time - start_time

    #  print(f'Epoch [{epoch+1}/{num_epochs}]')
    #  print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    #  print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    #  print(f'Epoch Time: {epoch_time:.2f} seconds\n')

#  print('Training complete.')
#  def validate(model, criterion, dataloader, device):
    #  model.eval()  # Set model to evaluation mode
    #  running_loss = 0.0
    #  correct = 0
    #  total = 0

    #  with torch.no_grad():  # Disable gradient computation
        #  for images, labels in dataloader:
            #  images = images.to(device)
            #  labels = labels.to(device)

            #  # Forward pass
            #  outputs = model(images)
            #  loss = criterion(outputs, labels)

            #  # Statistics
            #  running_loss += loss.item() * images.size(0)
            #  _, predicted = torch.max(outputs.data, 1)
            #  total += labels.size(0)
            #  correct += (predicted == labels).sum().item()

    #  epoch_loss = running_loss / total
    #  epoch_acc = 100 * correct / total

    #  return epoch_loss, epoch_acc
#  import matplotlib.pyplot as plt

#  # Plot loss
#  plt.figure(figsize=(10, 5))
#  plt.plot(train_losses, label='Training Loss')
#  plt.plot(val_losses, label='Validation Loss')
#  plt.title('Loss Over Epochs')
#  plt.xlabel('Epoch')
#  plt.ylabel('Loss')
#  plt.legend()
#  plt.show()

#  # Plot accuracy
#  plt.figure(figsize=(10, 5))
#  plt.plot(train_accuracies, label='Training Accuracy')
#  plt.plot(val_accuracies, label='Validation Accuracy')
#  plt.title('Accuracy Over Epochs')
#  plt.xlabel('Epoch')
#  plt.ylabel('Accuracy (%)')
#  plt.legend()
#  plt.show()
#  # Load the best model weights
#  model.load_state_dict(torch.load('best_model.pth'))

#  # Evaluate on test set
#  test_loss, test_acc = validate(model, criterion, test_loader, device)
#  print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
#  from sklearn.metrics import classification_report, confusion_matrix
#  import numpy as np

#  model.eval()
#  all_labels = []
#  all_preds = []

#  with torch.no_grad():
    #  for images, labels in test_loader:
        #  images = images.to(device)
        #  labels = labels.to(device)
        #  outputs = model(images)
        #  _, predicted = torch.max(outputs.data, 1)
        #  all_labels.extend(labels.cpu().numpy())
        #  all_preds.extend(predicted.cpu().numpy())

#  # Generate classification report
#  print(classification_report(all_labels, all_preds, target_names=['Genuine', 'Forged']))

#  # Compute confusion matrix
#  cm = confusion_matrix(all_labels, all_preds)
#  print('Confusion Matrix:')
#  print(cm)
#  # Assuming all_labels and all_preds are already defined
#  import matplotlib.pyplot as plt

#  misclassified_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]

#  # Get some misclassified samples
#  num_samples = 5  # Number of samples to display
#  misclassified_samples = misclassified_indices[:num_samples]

#  for idx in misclassified_samples:
    #  image, label = test_dataset[idx]
    #  image = image.to(device)
    #  output = model(image.unsqueeze(0))
    #  _, predicted = torch.max(output.data, 1)

    #  plt.imshow(image.cpu().permute(1, 2, 0).squeeze(), cmap='gray')
    #  plt.title(f'Actual: {label}, Predicted: {predicted.item()}')
    #  plt.axis('off')
    #  plt.show()
#  torch.save(model.state_dict(), 'final_model.pth')

#  # Load the best model weights
#  model.load_state_dict(torch.load('best_model.pth'))

#  # Set the model to evaluation mode
#  model.eval()
#  def test_model(model, criterion, dataloader, device):
    #  model.eval()  # Set model to evaluation mode
    #  running_loss = 0.0
    #  correct = 0
    #  total = 0

    #  with torch.no_grad():  # Disable gradient computation
        #  for images, labels in dataloader:
            #  images = images.to(device)
            #  labels = labels.to(device)

            #  # Forward pass
            #  outputs = model(images)
            #  loss = criterion(outputs, labels)

            #  # Accumulate loss
            #  running_loss += loss.item() * images.size(0)

            #  # Calculate predictions
            #  _, predicted = torch.max(outputs.data, 1)
            #  total += labels.size(0)
            #  correct += (predicted == labels).sum().item()

    #  # Calculate loss and accuracy
    #  test_loss = running_loss / total
    #  test_acc = 100 * correct / total

    #  return test_loss, test_acc
#  # Evaluate on the test set
#  test_loss, test_acc = test_model(model, criterion, test_loader, device)

#  print(f"Test Loss: {test_loss:.4f}")
#  print(f"Test Accuracy: {test_acc:.2f}%")
#  from sklearn.metrics import classification_report

#  # Collect all predictions and labels
#  all_preds = []
#  all_labels = []

#  model.eval()
#  with torch.no_grad():
    #  for images, labels in test_loader:
        #  images = images.to(device)
        #  labels = labels.to(device)
        #  outputs = model(images)
        #  _, predicted = torch.max(outputs.data, 1)
        #  all_preds.extend(predicted.cpu().numpy())
        #  all_labels.extend(labels.cpu().numpy())

#  # Generate classification report
#  target_names = ['Genuine', 'Forged']
#  print(classification_report(all_labels, all_preds, target_names=target_names))
#  from sklearn.metrics import confusion_matrix
#  import seaborn as sns
#  import matplotlib.pyplot as plt

#  # Compute confusion matrix
#  cm = confusion_matrix(all_labels, all_preds)

#  # Plot the confusion matrix
#  plt.figure(figsize=(8, 6))
#  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
#  plt.xlabel('Predicted')
#  plt.ylabel('Actual')
#  plt.title('Confusion Matrix')
#  plt.show()
#  # Identify misclassified samples
#  misclassified_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_preds)) if label != pred]

#  # Visualize a few misclassified examples
#  num_samples = 5
#  for idx in misclassified_indices[:num_samples]:
    #  img, label = test_dataset[idx]
    #  img = img.squeeze().numpy()  # Convert to NumPy array
    #  plt.imshow(img, cmap='gray')
    #  plt.title(f"Actual: {target_names[label]}, Predicted: {target_names[all_preds[idx]]}")
    #  plt.axis('off')
    #  plt.show()
