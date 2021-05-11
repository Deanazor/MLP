from optimizers import SGD
from layers import Dense, Output
from activations import relu, softmax
import numpy as np

class Model:    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.compiled = True

class MLP():
    def __init__(self, input_shape, output_shape):
        self.dense1 = Dense(input_shape, 2)
        self.activation1 = relu()
        self.dense2 = Dense(2, output_shape)
        self.activation2 = Output()
        self.optimizer = SGD(lr=0.1, momentum=0.1)

    def forward(self, X):
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.outputs)

        self.dense2.forward(self.activation1.outputs)
    
    def backward(self, y):
        self.activation2.backward(self.activation2.outputs, y)
        self.dense2.backward(self.activation2.dinputs)

        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)

    def optimize(self):
        self.optimizer.update_params(self.dense1)
        self.optimizer.update_params(self.dense2)
    
    def train(self, X, y, epochs=10):
            
        for i in range(epochs):
            # Forward Propagation / Feed Forward
            self.forward(X)
            loss = self.activation2.forward(self.dense2.outputs, y)

            preds = np.argmax(self.activation2.outputs, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(preds == y)

            print("epoch {}/{}: loss: {}; acc: {}".format(i+1, epochs, loss, accuracy))

            # Back Propagation / Backward pass
            self.backward(y)
            self.optimize()

    