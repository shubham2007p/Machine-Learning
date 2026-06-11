# Multiple Linear Regression From Scratch using Gradient Descent

My final implementation of Linear Regression from scratch before moving to ML libraries and real datasets.

## Goal

Build a Machine Learning model that can learn from multiple features and automatically optimize its parameters using Gradient Descent.

No scikit-learn, NumPy, TensorFlow, or PyTorch were used.

## Dataset

Features:

* CGPA
* Projects
* Communication Score

Label:

* Placement Score

Example:

```python
[8.5, 4, 7] -> 85
[7.2, 2, 5] -> 60
[9.1, 6, 9] -> 95
```

## Model

The model predicts placement score using multiple weighted features.

Prediction concept:

```text
Placement Score =
(CGPA × Weight₁)
+ (Projects × Weight₂)
+ (Communication × Weight₃)
+ Bias
```

## What I Learned

* Difference between single-feature and multi-feature regression.
* Every feature gets its own weight.
* Weights represent how strongly a feature influences the prediction.
* Machine Learning is fundamentally an optimization problem.
* Gradient Descent updates parameters to reduce prediction error.
* Learning Rate controls the size of parameter updates.
* Large learning rates can cause training instability.

## Gradient Descent Understanding

Training Loop:

```text
Predict
↓
Calculate Error
↓
Calculate Gradients
↓
Update Weights
↓
Repeat
```

The model gradually improves by reducing Mean Squared Error (MSE).

## Engineering Discovery

While building this project I encountered Gradient Explosion.

Using a large learning rate caused:

```text
Large Gradient
+
Large Learning Rate
=
Massive Parameter Jump
=
Increasing Error
```

Instead of moving toward the minimum error, the model repeatedly overshot the solution.

Reducing the learning rate stabilized training and allowed the model to converge.

## Developer Notes

This project helped me connect several Machine Learning concepts together:

* Features
* Labels
* Predictions
* Error
* MSE
* Derivatives
* Gradient Descent
* Multiple Parameters

The biggest realization was:

> Machine Learning is not about writing prediction formulas. It is about learning the correct parameters for those formulas.

## Learning Journey

Linear Regression V1
→ Brute Force Search

Linear Regression V2
→ Heuristic Optimization

Linear Regression V3
→ Single Feature Gradient Descent

Linear Regression V4
→ Multiple Features + Gradient Descent

## Next Step

* Vectors and Matrices
* NumPy
* Pandas
* Real-world Datasets
* scikit-learn Implementations

This project marks the completion of my Linear Regression foundations journey.
