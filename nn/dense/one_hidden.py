import numpy as np


# ACTIVATION FUNCTIONS

def sigmoid(x):
    """
    This is a sigmoid function. Sigmoids are important for machine learning
    because they always return a value between 0 and 1, which can be thought
    of as off or on.

    https://en.wikipedia.org/wiki/Sigmoid_function
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Back propagation uses the derivative of the activation function to adjust
    its weights.

    https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    """
    return x * (1 - x)  # NOTE: this is different from linear layers


# LOSS FUNCTION

def binary_crossentropy(y, y_pred):
    """
    https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
    """
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


# TRAINING FUNCTIONS

def training_loop(inputs, y, epochs, lr, iws, b1, hws, b2):
    hws_output = None
    for epoch in range(epochs):
        # forward pass
        iws_input = np.dot(inputs, iws) + b1
        iws_output = sigmoid(iws_input)
        hws_input = np.dot(iws_output, hws) + b2
        hws_output = sigmoid(hws_input)

        # compute loss
        loss = binary_crossentropy(y, hws_output)

        # backward pass
        delta_h = (hws_output - y) * sigmoid_derivative(hws_output)
        d_hws = np.dot(iws_output.T, delta_h)
        d_b2 = np.sum(delta_h, axis=0, keepdims=True)

        delta_i = np.dot(delta_h, hws.T) * sigmoid_derivative(iws_output)
        d_iws = np.dot(inputs.T, delta_i)
        d_b1 = np.sum(delta_i, axis=0, keepdims=True)

        # update weights and biases
        iws -= lr * d_iws
        b1 -= lr * d_b1
        hws -= lr * d_hws
        b2 -= lr * d_b2

        # progress update
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    return hws_output


# NETWORK

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # dataset
y = np.array([[0],    [1],    [1],    [0]])     # labels
input_layer_size = 2
hidden_layer_size = 4 
output_layer_size = 1
learning_rate = 0.1
epochs = 10000


# INITIALIZE WEIGHTS

def xavier_init(input_size, output_size):
    """
    Xavier initialization avoids the exploding / vanishing gradient problem
    better than random initialization. Training a network is faster as a
    result.

    https://pouannes.github.io/blog/initialization/
    """
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))


np.random.seed(42)
iws = xavier_init(input_layer_size, hidden_layer_size)
b1 = np.zeros((1, hidden_layer_size))
hws = xavier_init(hidden_layer_size, output_layer_size)
b2 = np.zeros((1, output_layer_size))


# TRAIN IT

hws_output = training_loop(X, y, epochs, learning_rate, iws, b1, hws, b2)
print("Final output after training:")
print(hws_output)

