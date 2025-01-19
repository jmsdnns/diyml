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


# LAYER

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = xavier_init(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input_data = input_data
        z = np.dot(self.input_data, self.weights) + self.biases
        self.output = sigmoid(z)
        return self.output

    def backward(self, d_output):
        d_activation = d_output * sigmoid_derivative(self.output)
        self.d_weights = np.dot(self.input_data.T, d_activation)
        self.d_biases = np.sum(d_activation, axis=0, keepdims=True)
        return np.dot(d_activation, self.weights.T)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases


# TRAINING LOOP

def training_loop(inputs, y, epochs, lr, layers):
    fwd_output = None

    for epoch in range(epochs):
        # forward pass
        activations = [inputs]
        for layer in layers:
            fwd_output = layer.forward(activations[-1])
            activations.append(fwd_output)

        # compute loss
        loss = binary_crossentropy(y, fwd_output)

        # backward pass
        d_output = fwd_output - y
        for layer in reversed(layers):
            d_output = layer.backward(d_output)

        # update weights and biases
        for layer in layers:
            layer.update(lr)

        # progress update
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return fwd_output


# NETWORK

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # dataset
y = np.array([[0],    [1],    [1],    [0]])     # labels
input_layer_size = 2
hidden_layer_sizes = [4, 4, 4]  # Three hidden layers with 4 neurons each
output_layer_size = 1
learning_rate = 1
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
layers = []
input_size = input_layer_size
for hidden_size in hidden_layer_sizes:
    layers.append(Layer(input_size, hidden_size))
    input_size = hidden_size
layers.append(Layer(input_size, output_layer_size))


# TRAIN IT

output = training_loop(X, y, epochs, learning_rate, layers)
print("Final output after training:")
print(output)

