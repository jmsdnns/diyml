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
    return sigmoid(x) * (1 - sigmoid(x))


# LOSS FUNCTION

def mean_squared_error(output, target):
    """
    MSE is a way of measuring the average difference between two values. The
    difference is squared to applying an increasingly harsher penalty for
    differences, which is a mathematical way of saying being close is ok but
    being far off is really, really bad.

    https://en.wikipedia.org/wiki/Mean_squared_error
    """
    return 0.5 * (output - target)**2  


# TRAINING FUNCTIONS

def forward_pass(x, hws1, hws2, hws3, ows):
    # 1st layer
    z1 = np.dot(x, hws1)
    a1 = sigmoid(z1)
    # 2nd layer
    z2 = np.dot(a1, hws2)
    a2 = sigmoid(z2)
    # 3rd layer
    z3 = np.dot(a2, hws3)
    a3 = sigmoid(z3)
    # output layer
    z4 = np.dot(a3, ows)
    output = sigmoid(z4)

    return output, (a1, a2, a3)


def backward_pass(x, output, a1, a2, a3):
    # output layer gradient
    delta_output = (output - target) * sigmoid_derivative(output)
    # 3rd layer backprop
    delta_hidden3 = np.dot(delta_output, ows.T) * sigmoid_derivative(a3)
    # 2nd layer backprop
    delta_hidden2 = np.dot(delta_hidden3, hws3.T) * sigmoid_derivative(a2)
    # 1st layer backprop
    delta_hidden1 = np.dot(delta_hidden2, hws2.T) * sigmoid_derivative(a1)

    # weight gradients
    dows = np.dot(a3.reshape(-1, 1), delta_output.reshape(1, -1))  # Gradient for ows
    dhws3 = np.dot(a2.reshape(-1, 1), delta_hidden3.reshape(1, -1))  # Gradient for hws3
    dhws2 = np.dot(a1.reshape(-1, 1), delta_hidden2.reshape(1, -1))  # Gradient for hws2
    dhws1 = np.dot(x.reshape(-1, 1), delta_hidden1.reshape(1, -1))  # Gradient for hws1

    return (dhws1, dhws2, dhws3, dows)


def update_weights(weights, gradients, lr):
    return [w - lr * dw for w,dw in zip(weights, gradients)]


def training_loop(inputs, epochs, lr, *weights):
    output = None
    for epoch in range(epochs):
        # forward pass
        output, activations = forward_pass(inputs, *weights)
        # compute error
        error = mean_squared_error(output, target)
        # backward pass
        gradients = backward_pass(inputs, output, *activations)
        # update weights
        weights = update_weights(weights, gradients, lr)
        # progress update
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Error: {error}")
    return output, weights


# NETWORK

target = np.array([1.0])  # target output
lr = 0.1                  # learning rate
input_size = 2
hidden_layer_sizes = [3, 3, 3]  # 3 hidden layers, each with 3 nodes
output_size = 1
epochs = 1000


# INITIALIZE WEIGHTS

## initial weights for input layer: 1x2
iws = np.array([0.1, 0.2])

## initial weights for hidden layer 1: 2x3
hws1 = np.random.rand(input_size, hidden_layer_sizes[0])
## initial weights for hidden layer 2: 3x3
hws2 = np.random.rand(hidden_layer_sizes[0], hidden_layer_sizes[1])
## initial weights for hidden layer 3: 3x3
hws3 = np.random.rand(hidden_layer_sizes[1], hidden_layer_sizes[2])

## initial weights for output layer: 3x1
ows = np.random.rand(hidden_layer_sizes[2], output_size)


# TRAIN IT
print("\n# TRAINING\n")
output, weights = training_loop(iws, epochs, lr, hws1, hws2, hws3, ows)

print("\n# RESULT\n")
print(f"- Output: {output}")
print(f"- Target: {target}\n")

print("# START WEIGHTS\n")
print(f"## hws1:\n{hws1}\n")
print(f"## hws2:\n{hws2}\n")
print(f"## hws3:\n{hws3}\n")
print(f"## ows:\n{ows}\n")

print("# FINAL WEIGHTS\n")
print(f"## hws1:\n{weights[0]}\n")
print(f"## hws2:\n{weights[1]}\n")
print(f"## hws3:\n{weights[2]}\n")
print(f"## ows:\n{weights[3]}\n\n")

