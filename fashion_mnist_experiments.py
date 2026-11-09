import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = (
    fashion_mnist.load_data()
)

label_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Normalize pixel values to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Plot sample grid
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(label_names[train_labels[i]])
plt.show()

# --- 2a: Sigmoid + SGD (10 Epochs) ---
model_2a = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model_2a.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model_2a.fit(train_images, train_labels, epochs=10)
test_loss_2a, test_acc_2a = model_2a.evaluate(test_images, test_labels)
print("2a Test Accuracy (Sigmoid, SGD, 10 epochs):", test_acc_2a)

# --- 2b: Sigmoid + SGD (100 Epochs) ---
model_2b = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model_2b.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model_2b.fit(train_images, train_labels, epochs=100)
test_loss_2b, test_acc_2b = model_2b.evaluate(test_images, test_labels)
print("2b Test Accuracy (Sigmoid, SGD, 100 epochs):", test_acc_2b)

# --- 2c: ReLU + Adam (10 Epochs) ---
model_2c = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model_2c.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model_2c.fit(train_images, train_labels, epochs=10)
test_loss_2c, test_acc_2c = model_2c.evaluate(test_images, test_labels)
print("2c Test Accuracy (ReLU, Adam, 10 epochs):", test_acc_2c)

# --- 2d: ReLU + Adam (100 Epochs) ---
model_2d = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model_2d.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model_2d.fit(train_images, train_labels, epochs=100)
test_loss_2d, test_acc_2d = model_2d.evaluate(test_images, test_labels)
print("2d Test Accuracy (ReLU, Adam, 100 epochs):", test_acc_2d)
