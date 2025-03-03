import os
import shutil
import random

# Define the source directory for the Bengali dataset
# huajiwang
source_dir = "/content/BHSig260-Bengali"

# Define the ratios for splitting
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Define the base directory for the split data
base_dir = "data"

# Define the splits and classes
splits = ["train", "val", "test"]
classes = ["genuine", "forged"]

# Create the directory structure if it doesn't exist
for split in splits:
    for cls in classes:
        folder = os.path.join(base_dir, split, cls)
        os.makedirs(folder, exist_ok=True)

# Loop through each person's directory in the source directory
for person_dir in os.listdir(source_dir):
    person_path = os.path.join(source_dir, person_dir)

    # Get a list of all signature files for this person
    signature_files = os.listdir(person_path)
    num_signatures = len(signature_files)

    # Calculate the number of signatures for each split
    num_train = int(train_ratio * num_signatures)
    num_val = int(val_ratio * num_signatures)
    num_test = num_signatures - num_train - num_val

    # Shuffle the signature files randomly
    random.shuffle(signature_files)

    # Split the signature files into train, val, and test sets
    train_files = signature_files[:num_train]
    val_files = signature_files[num_train:num_train + num_val]
    test_files = signature_files[num_train + num_val:]

    # Copy the files to the corresponding folders
    for file in train_files:
        src_path = os.path.join(person_path, file)
        dst_path = os.path.join(base_dir, "train", "genuine" if "G" in file else "forged", file)
        shutil.copy(src_path, dst_path)

    for file in val_files:
        src_path = os.path.join(person_path, file)
        dst_path = os.path.join(base_dir, "val", "genuine" if "G" in file else "forged", file)
        shutil.copy(src_path, dst_path)

    for file in test_files:
        src_path = os.path.join(person_path, file)
        dst_path = os.path.join(base_dir, "test", "genuine" if "G" in file else "forged", file)
        shutil.copy(src_path, dst_path)

print("Dataset split and copied successfully!")

for split in ["train", "val", "test"]:
    for cls in ["genuine", "forged"]:
        folder_path = os.path.join(base_dir, split, cls)
        num_files = len(os.listdir(folder_path))
        print(f"Number of files in {split}/{cls}: {num_files}")
