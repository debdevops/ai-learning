# Week 7: Tuning Deep Learning Models

### Objective
To move beyond building a basic neural network and learn the essential techniques for diagnosing, improving, and optimizing its performance. We will learn how to visualize the training process, tune key hyperparameters, and use callbacks to save our best model and prevent overfitting.

### Key Concepts

1.  **Visualizing Training History:**
    *   The `.fit()` method in Keras returns a `History` object. This object contains the loss and accuracy metrics for each epoch on both the training and validation sets.
    *   By plotting these metrics, we can create **learning curves**. These plots are the single most important tool for diagnosing a model's health. They can tell us if the model is learning effectively, if it's underfitting, or if it's overfitting.

2.  **Hyperparameter Tuning:**
    *   **Hyperparameters** are the "settings" of the model that we, the developers, choose *before* training begins (e.g., the number of layers, the number of neurons, the type of optimizer).
    *   **Learning Rate:** One of the most critical hyperparameters. It controls how large the "steps" are during Gradient Descent. Finding a good learning rate is crucial for effective training.

3.  **Callbacks:**
    *   Callbacks are tools that can be "called" by Keras at different points during the training process (e.g., at the end of each epoch).
    *   **`ModelCheckpoint`:** A callback that saves the model's weights to a file, but only when its performance on the validation set improves. This ensures you always have a copy of the **best version** of your model from the entire training run.
    *   **`EarlyStopping`:** A callback that monitors a specific metric (like validation loss). If that metric stops improving for a specified number of epochs (the `patience`), it will automatically stop the training. This is the best way to prevent overfitting and save unnecessary training time.

### Files in this folder
- `README.md`: This file.
- `neural_network_tuning.py`: A script that builds upon our Week 6 fashion classifier, demonstrating how to plot learning curves and use callbacks for better training. 