# --- Step 0: Imports ---
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow and other libraries imported.")

# --- Step 1: Load the Fashion MNIST Dataset ---
# Keras provides this dataset directly, just like the digit MNIST.
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(f"\nFashion MNIST dataset loaded. Shape of training data: {X_train_full.shape}")

# To understand our labels, we need to know what each number corresponds to.
# Let's define the class names. The label '0' is 'T-shirt/top', '1' is 'Trouser', and so on.
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# --- Step 2: Data Preparation (Identical to before) ---
# We normalize the pixel values to be between 0 and 1.
# And we create a validation set.
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0


# --- Step 3: Build the Neural Network Architecture (Identical to before) ---
# The architecture that worked for digits is a great starting point for this problem too.
# The input shape is the same (28x28), and the output classes are the same (10).
print("\nBuilding the Neural Network model...")
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()


# --- Step 4: Compile the Model (Identical to before) ---
print("\nCompiling the model...")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# --- Step 5: Train the Model (Identical to before) ---
# We fit the model to our new fashion data.
print("\nTraining the model on the Fashion MNIST data...")
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))


# --- Step 6: Evaluate the Model (Identical to before) ---
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")


# --- Step 7: Make and Visualize Predictions ---
# Let's take the first 3 images from the test set and predict what they are.
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_pred = np.argmax(y_proba, axis=1)

print("\n--- Making predictions on new images ---")
print(f"Predicted classes (as numbers): {y_pred}")

# Let's convert the predicted numbers back to their text labels using our class_names list.
predicted_labels = np.array(class_names)[y_pred]
actual_labels = np.array(class_names)[y_test[:3]]

print(f"Predicted labels (as text):   {predicted_labels}")
print(f"Actual labels (as text):      {actual_labels}")

# Visualize the first prediction
plt.figure(figsize=(8, 8))
plt.imshow(X_test[0], cmap='binary')
plt.title(f"Model Prediction: {predicted_labels[0]}", fontsize=16)
plt.axis('off')
plt.show()