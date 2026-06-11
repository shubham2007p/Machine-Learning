# Linear Regression From Scratch (Optimized Search Version)

My second implementation of Linear Regression from scratch in Python.

## Why I Built This

My first version used brute force and tested every possible slope and intercept combination.

In this version, I wanted to make the model smarter by reducing unnecessary search and using information from the dataset itself.

## What Changed

### Version 1

* Tested many `(m, b)` combinations.
* Kept the combination with lowest MSE.
* Worked, but inefficient.

### Version 2

* Estimated an initial slope from the dataset.
* Optimized slope and intercept separately.
* Moved parameters only when MSE improved.
* Reached the same final model with far fewer searches.

## What I Learned

* A model is just parameters and rules.
* Training means reducing prediction error.
* MSE can guide parameter updates.
* Good initial guesses can reduce search time.
* Optimization is often more important than prediction code.

## Engineering Thoughts

While building this, I realized that the hard part of Machine Learning is not prediction:

```python
prediction = (m * x) + b
```

The hard part is:

* How to find good parameters.
* How to reduce error efficiently.
* How to decide which parameter should change.

This project made me think more about optimization than coding.

## Results

Best Model Found:

```text
m = 9
b = 34
MSE = 1.1667
```

Prediction Formula:

```text
y = 9x + 34
```
