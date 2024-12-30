import numpy as np
import matplotlib.pyplot as plt


def generate_coffee_data(n_samples=100):
    # Light roast beans (shorter time, lower temp)
    light_roast = np.random.normal(loc=[12, 410], scale=[1, 10], size=(n_samples//2, 2))
    # Dark roast beans (longer time, higher temp)
    dark_roast = np.random.normal(loc=[15, 440], scale=[1, 10], size=(n_samples//2, 2))

    X = np.vstack([light_roast, dark_roast])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])

    # randomize X and y the same way
    idxs = np.random.permutation(len(y))
    X = X[idxs]
    y = y[idxs]

    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, weights):
    return sigmoid(np.dot(X, weights))


def binary_cross_entropy_loss(y_true, y_pred):
    m = len(y_true)
    epsilon = 1e-10  # not obvious, but this avoids log(0)
    return -(1/m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))


def compute_gradients(X, y_true, y_pred):
    m = len(y_true)
    return (1/m) * np.dot(X.T, (y_pred - y_true))


def train(X, y, weights, learning_rate=0.1, num_iterations=1000):
    for i in range(num_iterations):
        # make estimate
        y_pred = predict(X, weights)
        # compute error
        loss = binary_cross_entropy_loss(y, y_pred)
        # adjust the weights
        gradients = compute_gradients(X, y, y_pred)
        weights -= learning_rate * gradients
        # print progress
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    
    return weights


# INITIALIZE ##########################

## 2D data
X, y = generate_coffee_data(n_samples=200)
## normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std
## add bias to inputs
X_bias = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

## randomize weights
weights = np.random.randn(X_bias.shape[1])

# TRAIN IT ############################
weights = train(X_bias, y, weights, learning_rate=0.1, num_iterations=1000)

# TEST IT #############################
y_pred = predict(X_bias, weights)
predictions = (y_pred >= 0.5).astype(int)
accuracy = np.mean(predictions == y)
## log results
print("\nPredictions:")
for prediction, yi, (time, temp) in zip(predictions, y, X):
    print("{} {} {:.4f} {:.4f}".format(prediction, int(yi), time, temp))
print(f"Accuracy: {accuracy * 100:.2f}%")

# PLOT IT #############################
plt.figure(figsize=(10, 6))
## light roast (0) in blue
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Light Roast', alpha=0.6)
## dark roast (1) in red
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Dark Roast', alpha=0.6)
## grid of points for decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
## compute decision boundary (sigmoid output = 0.5)
Z = sigmoid(np.dot(np.c_[np.ones(xx.ravel().shape), (np.vstack([xx.ravel(), yy.ravel()]).T - X_mean) / X_std], weights))
Z = Z.reshape(xx.shape)
## plot decision boundary
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')
## plot the results
plt.title('Coffee Roast Classification with Logistic Regression')
plt.xlabel('Roast Time')
plt.ylabel('Roast Temperature')
plt.legend()
plt.grid(True)
plt.show()
