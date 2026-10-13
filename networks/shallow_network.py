
import numpy as np

#THE NETWORK'S WEIGHTS ARE OPTIMISED WITH FULL BATCH GRADIENT DESCEINT WITH LEARNING RATE TAU

class ShallowNetwork:
    def __init__(self, in_dim, out_dim, init_w = None):

        if init_w is not None:
            self.W = init_w.copy()
        else:
            self.W = np.random.randn(out_dim, in_dim)


    def forward(self, x): 
        self.z = self.W @ x
        return self.z

    def backward(self, x, y, learning_rate):

        # self.forward(x)

        forward = self.W @ x

        dW = 1/x.shape[1] * (forward-y) @ x.T 



        self.W -= learning_rate * dW


    def train(self, X_train, Y_train, epochs, learning_rate):
        ws = []
        # print('X_train shape: ', X_train.shape)
        # print('Y_train shape: ', Y_train.shape)
        losses = []
        # print('epochs: ', epochs)
        for _ in range(epochs):
            # loss = 1/(2*X_train.shape[1]) * np.linalg.norm((W))
            loss = 1/X_train.shape[1] * 0.5 * np.linalg.norm(self.forward(X_train) - Y_train)**2
            # loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            ws.append(self.W.copy())
            self.backward(X_train, Y_train, learning_rate)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        return ws, losses
    