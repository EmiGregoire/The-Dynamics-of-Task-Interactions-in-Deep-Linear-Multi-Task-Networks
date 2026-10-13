import numpy as np

# This class defines a simple Tanh-based neural network with one hidden layer.
# It includes methods for forward propagation, backward propagation (gradient descent),
# and training the network. The class can be initialized with custom weights or with random
# weights if none are provided. It uses the Tanh activation function for the hidden layer.

class TanhNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1=None, init_w2=None):
        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def forward(self, x):
        self.a1 = self.W1 @ x
        self.z1 = self.tanh(self.a1)
        self.z2 = self.W2 @ self.z1
        return self.z2
    
    def backward(self, x, y, learning_rate):
        m = x.shape[1]
        
        # Forward pass
        self.forward(x)
        
        # Backward pass
        dz2 = (self.z2 - y) / m
        dW2 = dz2 @ self.z1.T
        
        dz1 = (self.W2.T @ dz2) * self.tanh_derivative(self.a1)
        dW1 = dz1 @ x.T
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1
    
    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        losses = []
        
        for epoch in range(epochs):
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
            # Uncomment to print loss every epoch
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        
        return w1s, w2s, losses
        
        return w1s, w2s, losses


class MTL_TanhNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim_1, out_dim_2,  init_ws = None, init_w1 = None, init_w2=None):
        if init_ws is not None and init_w1 is not None and init_w2 is not None:
            self.WS = init_ws.copy()
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.WS = np.random.randn(hidden_dim, in_dim)
            self.W1 = np.random.randn(out_dim_1, hidden_dim)
            self.W2 = np.random.randn(out_dim_2, hidden_dim)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        self.z = self.WS @ x
        self.z1 = self.tanh(self.z)
        self.T1 = self.W1 @ self.z1
        self.T2=self.W2@self.z1
        return self.T1, self.T2

    def backward(self, x, y, learning_rate):
        m = x.shape[1]

        # Forward pass
        self.forward(x)

        # Backward pass
        dz2 = (self.z2 - y) / m
        dW2 = dz2 @ self.z1.T

        dz1 = (self.W2.T @ dz2) * self.tanh_derivative(self.a1)
        dW1 = dz1 @ x.T

        # Update weights
        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1

    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        losses = []

        for epoch in range(epochs):
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
            # Uncomment to print loss every epoch
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        return w1s, w2s, losses

        return w1s, w2s, losses

