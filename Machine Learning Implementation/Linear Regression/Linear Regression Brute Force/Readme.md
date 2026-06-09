# Linear Regression From Scratch

My first machine learning model built from scratch using only Python.

## Why I Built This

I wanted to understand what a machine learning model actually is before using libraries like scikit-learn.

Instead of calling pre-built functions, I implemented the core ideas myself.

## What I Learned

* Features and labels
* Prediction using `y = mx + b`
* Model parameters (`m` and `b`)
* Mean Squared Error (MSE)
* Training vs prediction
* How a model searches for better parameters
* Why minimizing error is the core of learning

## My Approach

The model:

1. Stores slope and intercept.
2. Predicts outputs using a line.
3. Calculates prediction error using MSE.
4. Tries different `(m, b)` combinations.
5. Keeps the combination with the lowest error.

No ML libraries were used.

## Developer Notes

A few things I struggled with while building:

* Understanding how functions outside a class can use class methods.
* Passing model objects between functions.
* Designing the training flow.
* Deciding where the "best model" should be stored.
* Separating prediction, evaluation, and training logic.

The biggest realization was that training is basically:

Predict → Measure Error → Keep Better Parameters

## Current Result

Learned Model:

`y = 9x + 34`

Training MSE:

`1.1667`

## Next Step

Replace brute-force search with Gradient Descent and make the model learn more efficiently.
