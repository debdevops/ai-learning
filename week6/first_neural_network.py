# --- Step 0: Imports ---
# Import TensorFlow and the Keras API, its high-level interface.
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow and other libraries imported successfully.")

# --- Step 1: Load the Dataset ---
# The MNIST dataset is so famous, it's built directly into Keras.
# This one line downloads the data and splits it into training and testing sets.
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print(f"\nDataset loaded. Shape of training data: {X_train_full.shape}")
print(f"Number of training labels: {len(y_train_full)}")


# --- Step 2: Data Preparation ---
# A) Normalize the pixel values.
# The pixel values range from 0 (black) to 255 (white).
# Neural networks work best with small input values, so we scale them to be between 0 and 1.
X_train_normalized = X_train_full / 255.0
X_test_normalized = X_test / 255.0

# B) Split the training data into a smaller training set and a validation set.
# The validation set is used during training to check the model's performance
# at the end of each training cycle (epoch).
X_valid, X_train = X_train_normalized[:5000], X_train_normalized[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


# --- Step 3: Build the Neural Network Architecture ---
# This is where we define the layers of our model. We'll use a simple sequential model.
print("\nBuilding the Neural Network model...")

model = keras.models.Sequential([
    # 1. Flatten Layer: This takes our 28x28 pixel image and flattens it into a 1D array of 784 pixels.
    # This is the input layer for our network.
    keras.layers.Flatten(input_shape=[28, 28]),

    # 2. Hidden Layer 1: A dense (fully connected) layer with 300 neurons.
    # 'relu' (Rectified Linear Unit) is a common and effective activation function.
    keras.layers.Dense(300, activation="relu"),

    # 3. Hidden Layer 2: Another dense layer with 100 neurons.
    keras.layers.Dense(100, activation="relu"),

    # 4. Output Layer: A dense layer with 10 neurons, one for each possible digit (0-9).
    # 'softmax' activation is crucial for multi-class classification. It ensures the
    # output of all 10 neurons sums to 1, giving us a probability for each class.
    keras.layers.Dense(10, activation="softmax")
])

# Print a summary of the model's architecture
model.summary()


# --- Step 4: Compile the Model ---
# Here we configure the training process.
print("\nCompiling the model...")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", # Stochastic Gradient Descent optimizer
              metrics=["accuracy"])


# --- Step 5: Train the Model ---
# This is the main training step where the model learns from the data.
# An 'epoch' is one full pass through the entire training dataset.
print("\nTraining the model...")
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))


# --- Step 6: Evaluate the Model ---
# We use the test set (which the model has never seen) to get a final, unbiased
# evaluation of its performance.
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(X_test_normalized, y_test)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")


# --- Step 7: Make a Prediction on a New Image ---
# Let's take the first 3 images from the test set and see what our model predicts.
X_new = X_test_normalized[:3]
y_proba = model.predict(X_new)

# The output is a list of probabilities for each class.
print("\n--- Making predictions on new images ---")
print("Probabilities for first 3 test images:")
print(y_proba.round(2))

# To get the final predicted class, we find the index of the highest probability.
y_pred = np.argmax(y_proba, axis=1)
print(f"\nPredicted classes: {y_pred}")
print(f"Actual classes:    {y_test[:3]}")

# Let's visualize the first prediction
plt.imshow(X_test[0], cmap='binary')
plt.title(f"Model Prediction: {y_pred[0]}")
plt.axis('off')
plt.show()