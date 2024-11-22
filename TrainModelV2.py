import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')

# Test if TensorFlow is utilizing the GPU for computation
import time

start = time.time()

# Data generator function
def data_generator(csv_file, base_dir, size=(200, 200), batch_size=32):
    """
    Data generator to load images in batches.
    :param csv_file: Path to the CSV file
    :param base_dir: Base directory for image files
    :param size: Image size (width, height)
    :param batch_size: Number of images to load per batch
    :return: Generator yielding (images, labels)
    """
    data = pd.read_csv(csv_file)
    num_samples = len(data)

    while True:  # Infinite loop for data generation
        for offset in range(0, num_samples, batch_size):
            batch_data = data.iloc[offset:offset + batch_size]
            images = []
            labels = []

            for _, row in batch_data.iterrows():
                img_path = os.path.join(base_dir, row['x:image'].strip())
                label = int(row['y:label'])

                # Load image
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, size).astype('float32') / 255.0
                    images.append(img)
                    labels.append(label)

            yield np.array(images), np.array(labels)


# Configure paths
train_binary_csv = './dataV2/binary/train_binary.csv'
test_binary_csv = './dataV2/binary/test_binary.csv'

binary_dir = './dataV2/binary'
rgb_dir = './dataV2/rgb'

# Total number of samples
train_samples = pd.read_csv(train_binary_csv).shape[0] * 2  # Combine grayscale and RGB
test_samples = pd.read_csv(test_binary_csv).shape[0] * 2

# Generators
train_generator_binary = data_generator(train_binary_csv, binary_dir, batch_size=32)
train_generator_rgb = data_generator(train_binary_csv, rgb_dir, batch_size=32)
test_generator_binary = data_generator(test_binary_csv, binary_dir, batch_size=32)
test_generator_rgb = data_generator(test_binary_csv, rgb_dir, batch_size=32)

# Combine generators
def combined_generator(generator1, generator2):
    while True:
        images1, labels1 = next(generator1)
        images2, labels2 = next(generator2)
        images = np.concatenate([images1, images2], axis=0)
        labels = np.concatenate([labels1, labels2], axis=0)
        yield images, labels


train_generator = combined_generator(train_generator_binary, train_generator_rgb)
test_generator = combined_generator(test_generator_binary, test_generator_rgb)

# Build the model
n_classes = 2  # Binary classification
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(200, 200, 3),
                        kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))  # Dropout

model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))

# Output layer
model.add(layers.Dense(n_classes))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Learning rate scheduler and early stopping
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Custom callback to print accuracy and loss for each epoch
class PrintEpochMetrics(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: "
              f"Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}, "
              f"Val_Loss = {logs['val_loss']:.4f}, Val_Accuracy = {logs['val_accuracy']:.4f}")


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_samples // 32,
    validation_data=test_generator,
    validation_steps=test_samples // 32,
    epochs=10,
    callbacks=[lr_scheduler, early_stopping, PrintEpochMetrics()]
)

# Save the model
model.save('person_V2.h5')
print("Model saved as person_V2.h5")
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# Plot training curves
def plot_accuracy_loss(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.show()


# Plot accuracy and loss curves
plot_accuracy_loss(history)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(test_generator, steps=test_samples // 32)
print(f"Validation accuracy: {val_accuracy:.2f}")
