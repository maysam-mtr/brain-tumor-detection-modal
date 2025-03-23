import os
from collections import Counter
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# dataset link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset (because size is too large)
# in my dataset however the data initially is not split into training and testing folders like the link,
# I only have one folder containing the total 4 classes

data = []
labels = []


# Paths to the folders for each class
glioma_path = "./brain-tumors-mris2/glioma"
meningioma_path = "./brain-tumors-mris2/meningioma"
pituitary_path = "./brain-tumors-mris2/pituitary"
no_tumor_path = "./brain-tumors-mris2/notumor"


def preprocess_image(img):
    try:
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply brightness normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized_img = clahe.apply(gray_img)

        # Detect the brain region (assumes center focus)
        height, width = normalized_img.shape
        center_x, center_y = width // 2, height // 2
        crop_size = min(height, width) // 2
        cropped_img = normalized_img[center_y - crop_size:center_y + crop_size,
                                     center_x - crop_size:center_x + crop_size]

        #Resize
        processed_img = cv2.resize(cropped_img, (128, 128))

        # Convert back to 3-channel
        return cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

def load_images_with_preprocessing(path, label):
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            # Load and preprocess the image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            img = preprocess_image(img)
            if img is None:  # If preprocessing fails, skip the image
                continue

            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

# Use the new function for loading and preprocessing images
load_images_with_preprocessing(glioma_path, 0)
load_images_with_preprocessing(meningioma_path, 1)
load_images_with_preprocessing(pituitary_path, 2)
load_images_with_preprocessing(no_tumor_path, 3)

data = np.array(data)
labels = np.array(labels)
data = data / 255.0  # Normalize pixel values to [0, 1]

label_counts = Counter(labels)
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

plt.bar(class_names, [label_counts[i] for i in range(4)])  # Bar plot of class distribution
plt.title('Class Distribution')
plt.ylabel('Number of Samples')
plt.show()

_, unique_indices = np.unique(data, axis=0, return_index=True)
duplicates = len(data) - len(unique_indices)
if duplicates > 0:
    print(f"Removing {duplicates} duplicate images.")
    data = data[unique_indices]
    labels = labels[unique_indices]

plt.figure(figsize=(12, 12))
for i in range(16):  # Show 16 random samples
    idx = np.random.randint(0, len(data))
    plt.subplot(4, 4, i + 1)
    plt.imshow(data[idx])
    plt.title(class_names[labels[idx]])
    plt.axis('off')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train, 4)
y_test = tf.keras.utils.to_categorical(y_test, 4)

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model
model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

loss, accuracy = model.evaluate(X_test, y_test)  # Evaluate on test data
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], 'r', label="Training Loss")
plt.plot(history.history['val_loss'], 'b', label="Validation Loss")
plt.legend(loc='upper left')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)  # Plot confusion matrix
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.show()
