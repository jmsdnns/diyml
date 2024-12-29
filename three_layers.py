import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mean_squared_error(output, target):
    return 0.5 * (output - target)**2  


def forward_pass(x, w1, w2, w3, w4):
    # 1st layer
    z1 = np.dot(x, w1)
    a1 = sigmoid(z1)
    # 2nd layer
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)
    # 3rd layer
    z3 = np.dot(a2, w3)
    a3 = sigmoid(z3)
    # output layer
    z4 = np.dot(a3, w4)
    output = sigmoid(z4)

    return output, (a1, a2, a3)


def backward_pass(x, output, a1, a2, a3):
    # output layer gradient
    delta_output = (output - target) * sigmoid_derivative(output)
    # 3rd layer backprop
    delta_hidden3 = np.dot(delta_output, w4.T) * sigmoid_derivative(a3)
    # 2nd layer backprop
    delta_hidden2 = np.dot(delta_hidden3, w3.T) * sigmoid_derivative(a2)
    # 1st layer backprop
    delta_hidden1 = np.dot(delta_hidden2, w2.T) * sigmoid_derivative(a1)

    # weight gradients
    dw4 = np.dot(a3.reshape(-1, 1), delta_output.reshape(1, -1))  # Gradient for w4
    dw3 = np.dot(a2.reshape(-1, 1), delta_hidden3.reshape(1, -1))  # Gradient for w3
    dw2 = np.dot(a1.reshape(-1, 1), delta_hidden2.reshape(1, -1))  # Gradient for w2
    dw1 = np.dot(x.reshape(-1, 1), delta_hidden1.reshape(1, -1))  # Gradient for w1

    return (dw1, dw2, dw3, dw4)


def update_weights(weights, gradients, lr):
    return [w - lr * dw for w,dw in zip(weights, gradients)]


def training_loop(inputs, lr, *weights):
    output = None
    for epoch in range(1000):
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


# NETWORK #############################
x = np.array([0.1, 0.2])  # input layer
target = np.array([1.0])  # target output
lr = 0.1                  # learning rate
input_size = 2
hidden_layer_sizes = [3, 3, 3]  # 3 hidden layers, each with 3 nodes
output_size = 1

# INITIALIZE WEIGHTS ##################
## [[n, n, n], [n, n, n]]
w1 = np.random.rand(input_size, hidden_layer_sizes[0])
## [[n, n, n], [n, n, n], [n, n, n]]
w2 = np.random.rand(hidden_layer_sizes[0], hidden_layer_sizes[1])
## [[n, n, n], [n, n, n], [n, n, n]]
w3 = np.random.rand(hidden_layer_sizes[1], hidden_layer_sizes[2])
## [[n], [n], [n]]
w4 = np.random.rand(hidden_layer_sizes[2], output_size)

# TRAIN IT ############################
output, weights = training_loop(x, lr, w1, w2, w3, w4)


print("\nResult:")
print(f"Output: {output}")
print(f"Target: {target}")
print("\nInitial weights:")
print(f"w1:\n{w1}")
print(f"w2:\n{w2}")
print(f"w3:\n{w3}")
print(f"w4:\n{w4}")
print("\nFinal weights:")
print(f"w1:\n{weights[0]}")
print(f"w2:\n{weights[1]}")
print(f"w3:\n{weights[2]}")
print(f"w4:\n{weights[3]}")

