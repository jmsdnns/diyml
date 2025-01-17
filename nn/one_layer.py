import numpy as np


# NETWORK

## initial weights for input layer
iws = np.array([0.1, 0.2])

## initial weights for hidden layer
hws = np.array([[0.3, 0.4], [0.5, 0.6]])

## initial weights for output layer
ows = np.array([[0.7], [0.8]])

## train network towards this value
target = np.array([1.0])

## learning rate
lr = 0.1


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


# FORWARD PASS

hws_input = np.dot(iws, hws)            # [0.13, 0.16]
hws_output = sigmoid(hws_input)         # [0.53245431, 0.53991488]

ows_input = np.dot(hws_output, ows)     # 0.80464992
ows_output = sigmoid(ows_input)         # 0.69096826

error = 0.5 * (ows_output - target)**2  # 0.04775031


# BACKWARD PASS

d_o = (ows_output - target) * sigmoid_derivative(ows_input) # -0.06598789
d_h = np.dot(d_o, ows.T) * sigmoid_derivative(hws_input)    # [-0.01149923, -0.01311347]

d_ows = np.dot(hws_output.reshape(-1,1), d_o.reshape(-1,1)) # [[-0.03513554], [-0.03562785]]
d_hws = np.dot(iws.reshape(-1,1), d_h.reshape(1,-1))        # [[-0.00114992, -0.00131135],
                                                            #  [-0.00229985, -0.00262269]]


# UPDATE WEIGHTS

ows_new = ows - lr * d_ows  # [[0.70351355], [0.80356278]]

hws_new = hws - lr * d_hws  # [[0.30011499 0.40013113],
                            #  [0.50022998 0.60026227]]


# PRINT IT

print(f"# ERROR:\n{error}\n")

print("# WEIGHT UPDATES")
print(f"## OUTPUT:\n{d_ows}")
print(f"## HIDDEN:\n{d_hws}\n")

print("# NEW WEIGHTS")
print(f"## OUTPUT:\n{ows_new}")
print(f"## HIDDEN:\n{hws_new}\n")
