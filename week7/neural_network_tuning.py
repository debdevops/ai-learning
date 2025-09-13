# --- Step 0: Imports ---
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

print("TensorFlow and other libraries imported successfully.")

# --- Step 1: Load and Prepare the Data ---
# We load the Fashion MNIST dataset and prepare it exactly as we did in Week 6.
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values and create a validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

print("\nFashion MNIST data loaded and prepared.")


# --- Step 2: Build the Neural Network Architecture ---
# We use the same simple architecture as before.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# --- Step 3: Compile the Model ---
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# --- Step 4: Train the Model ---
# The key difference is that we now store the return value of model.fit()
# This 'history' object contains all the metrics from the training process.
print("\nTraining the model for 30 epochs...")
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
print("Model training complete.")


# --- Step 5: Visualize the Learning Curves ---
# This is the new, critical step. We will use the 'history' object to plot
# the model's performance over time.

print("\nVisualizing training history...")
# Create a DataFrame from the history dictionary
pd.DataFrame(history.history).plot(figsize=(12, 8))

# Set plot labels and grid
plt.grid(True)
plt.title('Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')

# Set the y-axis to start from 0 for better readability
plt.gca().set_ylim(0, 1)

# Show the plot
plt.show()

print("\nScript finished.")
