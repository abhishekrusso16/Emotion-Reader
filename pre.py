import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Root folder containing subfolders named 0 to 17 (emotion classes)
DATA_DIR = "D:/emotion reader/images"  # Fixed backslash issue

# Image resize dimensions
IMG_SIZE = 48

# Initialize data containers
images = []
labels = []

# Automatically map folder names (0, 1, ..., 17) as class labels
label_map = {}  # Maps class name to label index
label_counter = 0

# Traverse each class directory
for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    label_map[class_name] = label_counter
    label = label_counter
    label_counter += 1

    # Load and preprocess images
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

# Convert to numpy arrays and normalize
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
labels = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

# Output summary
print("✅ Data preprocessing complete!")
print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")
print(f"🗂️ Class label map: {label_map}")
