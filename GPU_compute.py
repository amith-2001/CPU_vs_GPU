import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp



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
    print("Could not compute on GPU. Running on CPU.")

# Train the model
start_time = time.time()
history = model.fit(train_data, epochs=5, validation_data=test_data)
end_time = time.time()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print("\nTest Accuracy:", test_acc)

# Get predicted probabilities for each class
y_pred_probs = model.predict(test_data)
y_true = np.concatenate([y for _, y in test_data])

# Binarize the labels
y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):  # Assuming 10 classes in CIFAR-10
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', linestyle='--', lw=2,
         label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-Class')
plt.legend(loc='lower right')
plt.show()

# Print the total training time
total_time = end_time - start_time
print(f"\nTotal Training Time: {total_time} seconds")
print("efficiency is :",total_time/num_gpus_available)