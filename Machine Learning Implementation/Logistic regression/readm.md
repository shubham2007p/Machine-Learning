# Logistic Regression Model for Student Placement Prediction

## Overview
This project implements a basic Logistic Regression model from scratch using NumPy to predict student placement based on several input features. The model is trained using Batch Gradient Descent and cross-entropy loss.

## Features
-   **Logistic Regression Implementation:** A custom `Logistic_regression_model` class that calculates predictions using the sigmoid function.
-   **Cross-Entropy Loss:** A `calculate_loss` function to evaluate the model's performance.
-   **Batch Gradient Descent:** A `train_model` function orchestrating the training process, calling `gradeint_descent` for weight updates.
-   **Predictive Inference:** The `main` function demonstrates how to train the model and then use it to predict placement for a new student based on user input.

## How to Run

1.  **Dependencies:** Ensure you have `numpy` installed. If not, you can install it using pip:
    ```bash
    pip install numpy
    ```

2.  **Code Structure:** The entire code is contained within a single Python script or notebook. It defines the model class, loss function, training function, gradient descent function, and the main execution logic.

3.  **Execute the Script:**
    Run the Python script. The `main` function will:
    a.  Initialize the Logistic Regression model with zero weights.
    b.  Train the model using the predefined `students` and `placed` datasets for 1000 epochs.
    c.  Print the loss and model weights (CGPA, project, communication, GitHub, internship) for each epoch, allowing you to observe the training progress.
    d.  After training, it will prompt you to enter metrics for a new student (CGPA, project count, communication score, GitHub contributions, internship count).
    e.  It will then predict the placement probability for this student and indicate whether they are likely to be placed (probability > 0.5) or not.

    ```bash
    python your_script_name.py
    ```
    or simply run the cells in your Jupyter/Colab notebook.

## Core Components

### `Logistic_regression_model` Class
-   `__init__(self, m_cgpa, m_project, m_comm, m_github, m_internship)`: Initializes the model's weights for each feature. These weights are learned during the training process.
-   `predict(self, student_metric)`: Takes a student's feature vector and returns the predicted probability of placement using the logistic sigmoid function.

### `calculate_loss(x_values, y_values, model)`
-   Calculates the average cross-entropy loss over the given dataset (`x_values`, `y_values`) for the current `model`. Includes numerical stability (epsilon clipping) for `np.log`.

### `train_model(x_values, y_values, model)`
-   Orchestrates the training loop for a specified number of `num_epochs` (default 1000).
-   In each epoch, it calls `gradeint_descent` to update the model's weights.
-   Prints the epoch number, current loss, and the updated weights to monitor training progress.

### `gradeint_descent(x_values, y_values, model)`
-   Implements one step of Batch Gradient Descent.
-   Calculates the gradient of the loss function with respect to each model weight by summing the error terms (`predicted_placement - actual_placement`) multiplied by the corresponding feature value across all training examples.
-   Updates the model's weights by subtracting a fraction (`learning_rate`) of the averaged gradients.

### Data
-   `students_input`: A list of lists, where each inner list represents a student with features: `[CGPA, project count, communication score, GitHub contributions, internship count]`.
-   `placed_output`: A list of binary values (1 for placed, 0 for not placed) corresponding to the `students_input`.
-   These are converted to NumPy arrays (`students`, `placed`) for efficient computation.

### `main(students, placed)` Function
-   The entry point for the script when executed.
-   Handles model initialization, training, and user interaction for making new predictions.

## Example Usage (from `main` function)

```python
# ... (after model training) ...

user_cgpa = float(input("Enter student's CGPA (e.g., 8.0): "))
user_project = float(input("Enter student's project count (e.g., 5): "))
user_comm = float(input("Enter student's communication score (e.g., 7): "))
user_github = float(input("Enter student's GitHub contributions (e.g., 10): "))
user_internship = float(input("Enter student's internship count (e.g., 1): "))

student_metric = np.array([user_cgpa , user_project , user_comm , user_github , user_internship])

predicted_placement_probability = model.predict(student_metric)

print(f"\nPredicted placement probability: {predicted_placement_probability:.4f}")
if predicted_placement_probability > 0.5:
  print("Based on the model, the student is likely to be placed.")
else:
  print("Based on the model, the student is less likely to be placed.")
```

## Limitations
-   This is a basic, custom implementation for educational purposes and may not be optimized for large datasets or complex scenarios.
-   It does not include regularization, feature scaling, or advanced optimization techniques (e.g., Adam, SGD with momentum).
-   The current `learning_rate` is fixed and might need tuning for different datasets.
-   The model is a simple linear classifier followed by a sigmoid activation; it may not capture non-linear relationships in data.
