#!/usr/bin/python
# -*- coding: utf-8 -*
##############################################################
# File Name: 5.verify_batch.py
# Author:
# mail:
# Created Time: Sun Mar  2 15:35:07 2025
# Brief:
##############################################################
import matplotlib.pyplot as plt

# Get one batch of training data
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Plot the first image in the batch
plt.imshow(images[0].permute(1, 2, 0).squeeze(), cmap="gray")  # Adjust for grayscale
plt.title(f"Label: {labels[0]}")
plt.axis("off")
plt.show()

print(f"Batch size: {images.size(0)}, Image shape: {images[0].shape}")
