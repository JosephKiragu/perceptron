# Perceptron Classifier

The Perceptron Classifier is a simple and foundational machine learning model based on linear binary classification. It is a single-layer neural network that can be used to classify linearly separable data. This project provides an implementation of the Perceptron algorithm in Python and demonstrates its usage with a simple example.

## Features

- Simple and easy-to-understand implementation of the Perceptron algorithm
- Support for custom learning rates and number of iterations
- Option to penalize (reduce) the learning rate after a specified number of iterations
- Training accuracy visualization with a learning curve plot

## Installation

There are no special installation requirements for this project. Simply clone the repository and ensure you have the required dependencies installed, including NumPy, Matplotlib, and scikit-learn.

```bash
git clone https://github.com/JosephKiragu/perceptron.git
cd perceptron
```

## Usage

You can use the `Perceptron` class in your own projects to train a perceptron model and make predictions on new data. To do so, simply import the class and follow the steps below:

1. Create an instance of the Perceptron class with desired parameters, such as learning rate and number of iterations.

```python
from perceptron import Perceptron

model = Perceptron(learning_rate=0.1, n_iters=1000, penalize=True)
```

2. Train the perceptron model using your training data.

```python
model.fit(x_train, y_train)
```

3. Make predictions on new, unseen data.

```python
predictions = model.predict(x_test)
```

You can also plot the learning curve during training to visualize the training accuracy over time.

## Dependencies

- Python 3.5 or higher
- NumPy
- Matplotlib
- scikit-learn
