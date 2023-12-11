import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Set to an empty string or '-1' to disable GPU

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import time


# Check the number of available GPUs
num_gpus_available = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpus_available)

# Load and preprocess the CIFAR-10 dataset
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

# Display information about the dataset
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Preprocess the data
def preprocess_image(image):
    image = tf.image.resize(image, (128, 128))  # Resize images to a consistent size
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image

# Apply preprocessing to the dataset
train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_data = train_data.map(lambda x, y: (preprocess_image(x), y))

test_data = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
test_data = test_data.map(lambda x, y: (preprocess_image(x), y))

# Shuffle and batch the data
train_data = train_data.shuffle(1000).batch(32)
test_data = test_data.batch(32)

# Use MirroredStrategy for data parallelism if GPUs are available
if num_gpus_available >= 1:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Define the CNN model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')  # Adjust the number of classes for CIFAR-10
        ])
        
        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
else:
    print("Running on CPU. No GPUs available.")
    # Define the CNN model (no MirroredStrategy for a single GPU or CPU)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # Adjust the number of classes for CIFAR-10
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(train_data, epochs=2, validation_data=test_data)
end_time = time.time()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print("\nTest Accuracy:", test_acc)

# Print the total training time
total_time = end_time - start_time
print(f"\nTotal Training Time: {total_time} seconds")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
