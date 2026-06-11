# Linear Regression From Scratch using Gradient Descent

My third implementation of Linear Regression from scratch in Python.

## Goal

After building a brute-force search version and an optimized search version, I wanted to implement the actual learning process used in Machine Learning: Gradient Descent.

No scikit-learn, NumPy, TensorFlow, or PyTorch were used.

## What This Model Does

The model:

1. Starts with an initial slope and intercept.
2. Makes predictions using `y = mx + b`.
3. Calculates Mean Squared Error (MSE).
4. Computes gradients for slope and intercept.
5. Updates parameters using a learning rate.
6. Repeats the process for multiple training cycles.

Over time the model learns parameters that reduce prediction error.

## What I Learned

* A model is just parameters.
* Training means optimizing parameters.
* MSE acts as a cost function.
* Gradients tell us how error changes when parameters change.
* Gradient Descent is repeated parameter improvement, not magic.

## My Understanding

Before this project I manually searched for good values of slope and intercept.

With Gradient Descent I no longer ask:

* Should slope increase?
* Should slope decrease?

The derivative already contains that information.

The training loop becomes:

Prediction → Error → Gradient → Update → Repeat

## Developer Notes

Building this project helped me understand that Machine Learning is mostly an optimization problem.

The prediction equation was easy:

```text
y = mx + b
```

The difficult part was figuring out how the model should improve itself.

Gradient Descent solved that by using the rate of change of the cost function to guide parameter updates.

## Features

* Linear Regression from scratch
* Mean Squared Error (MSE)
* Gradient Descent optimization
* Adjustable learning rate
* Training loop with multiple iterations
* Prediction on unseen input values

## Future Improvements

* Multiple Linear Regression
* Vectorized implementation
* NumPy version
* Train/Test evaluation
* Real-world datasets

## Learning Journey

Version 1:
Brute Force Search

↓

Version 2:
Heuristic Parameter Optimization

↓

Version 3:
Gradient Descent

This repository represents the point where parameter search became actual machine learning training.
