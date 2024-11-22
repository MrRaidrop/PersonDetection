import numpy as np
import pandas as pd
import cv2
import os
from xml.etree import ElementTree
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
keras = tf.keras

# Update class definitions, adding "no-person"
class_names = ['person', 'person-like', 'no-person']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

n_classes = 3  # Now there are 3 classes
size = (200, 200)  # Image size


def load_data():
    """
    Load datasets and preprocess images and labels.
    """
    datasets = ['Train/Train', 'Test/Test', 'Val/Val']
    output = []

    for dataset in datasets:
        imags = []
        labels = []
        directoryA = "data/" + dataset + "/Annotations"
        directoryIMG = "data/" + dataset + "/JPEGImages/"

        # Get and filter the file lists
        annotation_files = [f for f in os.listdir(directoryA) if f.endswith('.xml')]
        image_files = [f for f in os.listdir(directoryIMG) if f.endswith('.jpg')]

        # Sort the file lists
        annotation_files.sort()
        image_files.sort()

        # Check if the number of files is consistent
        if len(annotation_files) != len(image_files):
            print(f"Warning: Inconsistent number of Annotations and JPEGImages files in dataset {dataset}!")
            print(f"Annotations: {len(annotation_files)}, JPEGImages: {len(image_files)}")
            continue

        # Iterate through XML files and read the corresponding images
        for i, xml in enumerate(annotation_files):
            xmlf = os.path.join(directoryA, xml)

            # Check index bounds
            if i >= len(image_files):
                print(f"Warning: Index {i} exceeds the number of JPEGImages files.")
                break

            # Parse the XML file
            dom = ElementTree.parse(xmlf)
            vb = dom.findall('object')
            label = vb[0].find('name').text
            if label not in class_names_label:
                print(f"Warning: Unrecognized label '{label}'")
                continue

            labels.append(class_names_label[label])

            # Read and preprocess the image
            img_path = os.path.join(directoryIMG, image_files[i])
            curr_img = cv2.imread(img_path)

            if curr_img is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            curr_img = cv2.resize(curr_img, size)
            imags.append(curr_img)

        imags = np.array(imags, dtype='float32') / 255.0
        labels = np.array(labels, dtype='int32')
        output.append((imags, labels))

    return output


# Load datasets
(train_images, train_labels), (test_images, test_labels), (val_images, val_labels) = load_data()

# Check the shape of the datasets
print("Training set images:", train_images.shape)
print("Training set labels:", train_labels.shape)

from tensorflow.keras import regularizers
from tensorflow.keras import layers, models, regularizers, callbacks


# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(200, 200, 3),
                        kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))  # Add a Dropout layer

model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu',
                       kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))  # Add a Dropout layer

# Output layer, 3 nodes (3 classes)
model.add(layers.Dense(n_classes))

# Compile the model and adjust the learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Learning rate scheduler
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))

# Plot training curves
def plot_accuracy_loss(history):
    fig = plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label="acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label="val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss
    plt.subplot(222)
    plt.plot(history.history['loss'], 'bo--', label="loss")
    plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.show()

plot_accuracy_loss(history)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f"Validation set accuracy: {val_accuracy:.2f}")

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the model as person3.h5
model.save('person3.h5')
print("Model saved as person3.h5")
