# Week 6: Introduction to Neural Networks & Deep Learning

### Objective
To build our first Artificial Neural Networks (ANNs). This week marks our transition into Deep Learning, using the powerful TensorFlow and Keras libraries. We will solve two classic computer vision problems: classifying handwritten digits from the MNIST dataset and classifying fashion items from the Fashion MNIST dataset. This reinforces the core, reusable workflow of deep learning.

### Key Concepts

1.  **Deep Learning:**
    *   A subfield of machine learning based on Artificial Neural Networks, which are models inspired by the structure and function of the human brain. A "deep" model is one with multiple layers, allowing it to learn highly complex patterns.

2.  **Artificial Neural Network (ANN):**
    *   **Neuron:** The basic computational unit. It receives inputs, performs a simple calculation, and passes the result to the next layer.
    *   **Layers:** Neurons are organized into layers.
        *   **Input Layer (`Flatten`):** Receives the raw data (e.g., the 28x28 grid of pixels) and transforms it into a 1D array that the network can process.
        *   **Hidden Layers (`Dense`):** Intermediate layers where the model learns complex patterns. The more hidden layers, the "deeper" the network.
        *   **Output Layer (`Dense`):** Produces the final prediction (e.g., the probability for each of the 10 classes).
    *   **Activation Function:** A function applied to the output of each neuron.
        *   **`ReLU` (Rectified Linear Unit):** A common choice for hidden layers. It introduces non-linearity, allowing the network to learn much more complex relationships than a linear model.
        *   **`Softmax`:** The standard choice for the output layer in multi-class classification. It converts the raw scores from the final layer into a probability distribution, where all outputs sum to 1.

3.  **The Deep Learning Workflow with Keras:**
    *   **Model Definition (`Sequential`):** Stacking layers in sequence to build the network's architecture.
    *   **Compilation (`.compile()`):** Configuring the model for training by choosing:
        *   An **Optimizer** (e.g., `'sgd'`): The algorithm used to update the model's internal weights (an implementation of Gradient Descent).
        *   A **Loss Function** (e.g., `'sparse_categorical_crossentropy'`): The function that measures how wrong the model's predictions are (the "Cost Function").
    *   **Training (`.fit()`):** The process of feeding the model the training data for a specified number of cycles (**epochs**) to learn the patterns. A **validation set** is used during training to monitor performance on unseen data and check for overfitting.
    *   **Evaluation (`.evaluate()`):** Measuring the model's final, unbiased performance on the completely unseen test data.
    *   **Prediction (`.predict()`):** Using the trained model to get probability outputs for new data, which are then converted to final class predictions.

### Files in this folder
- `README.md`: This file, outlining the core theory of our first deep learning models.
- `first_neural_network.py`: A script that builds, compiles, trains, and evaluates our first neural network for classifying handwritten digits from the classic MNIST dataset.
- `fashion_classifier_nn.py`: A second project that applies the exact same neural network architecture and workflow to a slightly more challenging problem: classifying images of clothing from the Fashion MNIST dataset. This demonstrates the reusability of a deep learning workflow.```

This updated file now accurately documents everything you've accomplished this week. It highlights the core concepts and clearly explains the purpose of each script you've written.