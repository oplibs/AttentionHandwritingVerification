#!/usr/bin/python
# -*- coding: utf-8 -*
##############################################################
# File Name: 3.resize_images.py
# Author:
# mail:
# Created Time: Sun Mar  2 15:28:47 2025
# Brief:
##############################################################
from torchvision import transforms
from PIL import Image
import os

# Define resizing transformation
resize_transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# Loop through dataset folders
base_dir = "data"
for split in ["train", "val", "test"]:
    for cls in ["genuine", "forged"]:
        folder = os.path.join(base_dir, split, cls)
        for file in os.listdir(folder):
            if file.endswith(".png"):
                img_path = os.path.join(folder, file)
                img = Image.open(img_path)
                img_resized = resize_transform(img)
                img_resized.save(img_path)  # Overwrite with resized image

print("Resizing png completed!")

# Define resizing transformation
#  resize_transform = transforms.Compose([
    #  transforms.Resize((224, 224))
#  ])

# Loop through dataset folders
base_dir = "data"
for split in ["train", "val", "test"]:
    for cls in ["genuine", "forged"]:
        folder = os.path.join(base_dir, split, cls)
        for file in os.listdir(folder):
            if file.endswith(".tif"):
                img_path = os.path.join(folder, file)
                img = Image.open(img_path)
                img_resized = resize_transform(img)
                img_resized.save(img_path)  # Overwrite with resized image

print("Resizing tif completed!")
