import numpy as np


# NETWORK #############################

## input layer
x = np.array([0.1, 0.2])

## hidden layer
w1 = np.array([[0.3, 0.4], [0.5, 0.6]])

## output layer
w2 = np.array([[0.7], [0.8]])

## target output
target = np.array([1.0])

## learning rate
lr = 0.1

## activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## derivative of activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# FORWARD PASS ########################

## [0.1*0.3 + 0.2*0.5, 0.1*0.4 + 0.2*0.6] = [0.13, 0.16]
hidden_input = np.dot(x, w1)  

## hidden layer activation = [0.53245431, 0.53991488]
hidden_activation = sigmoid(hidden_input) 

## 0.53245431*0.7 + 0.53991488*0.8 = 0.80464992
output_input = np.dot(hidden_activation, w2)  

## output = 0.69096826
output = sigmoid(output_input)  

## error = 0.04775031
error = 0.5 * (output - target)**2  


# BACKWARD PASS #######################

## output layer gradients = -0.06598789
delta_output = (output - target) * sigmoid_derivative(output_input)

## hidden layer gradients = [-0.01149923, -0.01311347]
delta_hidden = np.dot(delta_output, w2.T) * sigmoid_derivative(hidden_input)

## weight gradients output->hidden = [[-0.03513554],
##                                    [-0.03562785]]
dw2 = np.dot(hidden_activation.reshape(-1,1), delta_output.reshape(-1,1))

## weight gradients hidden->input = [[-0.00114992, -0.00131135],
##                                   [-0.00229985, -0.00262269]]
dw1 = np.dot(x.reshape(-1,1), delta_hidden.reshape(1,-1))


# UPDATE WEIGHTS ######################

## w2_new = [[0.70351355],
##           [0.80356278]]
w2_new = w2 - lr * dw2

## w1_new = [[0.30011499 0.40013113],
##           [0.50022998 0.60026227]]
w1_new = w1 - lr * dw1


print(f"Initial error: {error}")
print("\nWeight updates:")
print(f"dw2:\n{dw2}")
print(f"dw1:\n{dw1}")
print("\nNew weights:")
print(f"w2_new:\n{w2_new}")
print(f"w1_new:\n{w1_new}")
