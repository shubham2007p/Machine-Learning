import numpy as np

# classes behaviour of model
class Logistic_regression_model():
  def __init__(self , m_cgpa , m_project , m_comm , m_github , m_internship):
    self.cgpa = m_cgpa
    self.project = m_project
    self.comm = m_comm
    self.github = m_github
    self.internship = m_internship

  def predict(self, student_metric):
    # Using a simple linear combination of features for prediction before sigmoid
    z = (self.cgpa * student_metric[0]) + \
        (self.project * student_metric[1]) + \
        (self.comm * student_metric[2]) + \
        (self.github * student_metric[3]) + \
        (self.internship * student_metric[4])
    return 1 / (1 + np.exp(-z))

# loss function, `cross-entropy loss or log loss function`
def calculate_loss(x_values , y_values , model):
  error = 0
  epsilon = 1e-10 # Small value to prevent log(0)
  for student , placement in zip(x_values, y_values):
    prediction = model.predict(student)
    # Clip predictions to avoid log(0) or log(1) issues
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    if placement:
      error += -np.log(prediction)
    else:
      error += -np.log(1 - prediction)
  return error / len(x_values)

# Training the model using Batch Gradient Descent
def train_model(x_values , y_values , model):
  num_epochs = 1000 # Number of training iterations
  
  print("Epoch | Loss       | CGPA      | Project   | Comm      | Github    | Internship")
  print("----------------------------------------------------------------------------------")

  for epoch in range(num_epochs):
    # Perform one step of gradient descent for the entire batch
    gradeint_descent(x_values , y_values , model)
    current_loss = calculate_loss(x_values , y_values , model)
    
    # Print progress
    print(f"{epoch:<5} | {current_loss:.6f} | {model.cgpa:.6f} | {model.project:.6f} | {model.comm:.6f} | {model.github:.6f} | {model.internship:.6f}")

# Gradient descent function
def gradeint_descent(x_values , y_values ,model):
  learning_rate = 0.1
  
  # Initialize sums for partial derivatives (gradients)
  sum_partial_derivative_cgpa = 0.0
  sum_partial_derivative_project = 0.0
  sum_partial_derivative_comm = 0.0
  sum_partial_derivative_github = 0.0
  sum_partial_derivative_internship = 0.0

  # Iterate through each student to accumulate gradients
  for student_features, actual_placement in zip(x_values, y_values):
    predicted_placement = model.predict(student_features)
    
    # The error term (y_hat - y) for logistic regression gradient
    error_term = predicted_placement - actual_placement

    # Accumulate partial derivatives for each weight
    sum_partial_derivative_cgpa += error_term * student_features[0]
    sum_partial_derivative_project += error_term * student_features[1]
    sum_partial_derivative_comm += error_term * student_features[2]
    sum_partial_derivative_github += error_term * student_features[3]
    sum_partial_derivative_internship += error_term * student_features[4]

  # Update model weights using the averaged gradients
  num_students = len(x_values)
  model.cgpa -= learning_rate * (sum_partial_derivative_cgpa / num_students)
  model.project -= learning_rate * (sum_partial_derivative_project / num_students)
  model.comm -= learning_rate * (sum_partial_derivative_comm / num_students)
  model.github -= learning_rate * (sum_partial_derivative_github / num_students)
  model.internship -= learning_rate * (sum_partial_derivative_internship / num_students)

# Data set
students_input = [
    [8.5, 4, 8, 12, 1],
    [7.0, 1, 5, 2, 0],
    [9.1, 6, 9, 20, 2],
    [6.5, 0, 4, 1, 0],
    [8.2, 5, 8, 15, 2],
    [5.9, 0, 3, 0, 0],
    [7.8, 3, 7, 10, 1],
    [7.2, 2, 5, 3, 0],
    [9.5, 7, 9, 25, 3],
    [6.8, 1, 6, 4, 0],
    [8.0, 4, 7, 11, 1],
    [6.2, 1, 4, 2, 0],
    [8.8, 5, 8, 18, 2],
    [7.1, 1, 5, 3, 0],
    [7.9, 3, 8, 14, 1],
    [5.5, 0, 2, 0, 0],
    [9.2, 6, 9, 22, 2],
    [7.4, 2, 6, 5, 0],
    [8.4, 4, 7, 13, 1],
    [6.9, 2, 5, 4, 0],
    [8.7, 4, 8, 16, 2],
    [6.1, 0, 4, 1, 0],
    [9.0, 5, 9, 19, 2],
    [7.3, 2, 6, 6, 0],
    [8.1, 3, 7, 12, 1],
    [6.4, 1, 5, 2, 0],
    [9.4, 6, 8, 24, 3],
    [6.7, 1, 4, 3, 0],
    [8.6, 5, 8, 17, 2],
    [7.5, 2, 6, 7, 0],
    [8.9, 6, 9, 21, 2],
    [6.0, 0, 3, 1, 0],
    [8.3, 4, 7, 14, 1],
    [7.0, 1, 5, 4, 0],
    [9.3, 7, 9, 26, 3],
    [6.6, 1, 5, 2, 0],
    [7.7, 3, 7, 9, 1],
    [5.8, 0, 3, 0, 0],
    [8.5, 4, 8, 15, 2],
    [7.2, 2, 6, 5, 0],
    [9.6, 7, 9, 30, 3],
    [6.3, 1, 4, 1, 0],
    [8.2, 3, 8, 11, 1],
    [7.4, 2, 5, 4, 0],
    [8.8, 5, 9, 20, 2],
    [6.8, 1, 4, 2, 0],
    [9.1, 6, 8, 22, 2],
    [7.1, 2, 5, 3, 0],
    [8.4, 4, 7, 13, 1],
    [6.5, 1, 4, 2, 0]
]

placed_output = [
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0
]
students = np.array(students_input)
placed = np.array(placed_output)

def main(students , placed):
  model = Logistic_regression_model(0.0, 0.0, 0.0, 0.0, 0.0) # Initialize weights to 0.0
  train_model(students , placed , model)
  while True:
    print("\n--- Predict for a new student ---")
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

    choice = input("want to continue ? ")
    if choice not in "yY":
      break

if __name__ == "__main__":
  main(students , placed)
