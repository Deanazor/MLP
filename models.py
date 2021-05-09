from layers import Dense
from activations import relu, softmax
import numpy as np

class MLP:
    def __init__(self, input_shape, output_shape):
        self.dense1 = Dense(input_shape, 128)
        self.activation1 = relu()
        self.dense2 = Dense(128,64)
        self.activation2 = relu()
        self.dense3 = Dense(64, output_shape)
        self.activation3 = softmax()
        self.compiled = False
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.compiled = True

    def forward(self, X):
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.outputs)

        self.dense2.forward(self.activation1.outputs)
        self.activation2.forward(self.dense2.outputs)

        self.dense3.forward(self.activation2.outputs)
        self.activation3.forward(self.dense3.outputs)
    
    def backward(self, y):
        self.loss.backward(self.activation3.outputs, y)
        self.activation3.backward(self.loss.dinputs)
        self.dense3.backward(self.activation3.dinputs)
        
        self.activation2.backwards(self.dense3.dinputs)
        self.dense2.backward(self.activation2.dinputs)

        self.activation1.backwards(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)

    def optimize(self):
        self.optimizer.update_params(self.dense1)
        self.optimizer.update_params(self.dense2)
        self.optimizer.update_params(self.dense3)

    def train(self, X, y, epochs=10):
        if not self.compiled:
            raise RuntimeError("Model is not compiled")
            
        for i in range(epochs):
            # Forward Propagation / Feed Forward
            self.forward(X)
            loss = self.loss.calc_loss(self.activation3.outputs, y)

            preds = np.argmax(self.activation3.outputs, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(preds == y)

            print("epoch {}/{}: loss: {}; acc: {}".format(i+1, epochs, loss, accuracy))

            # Back Propagation / Backward pass
            self.backward(y)
            self.optimize()