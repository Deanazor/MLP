import numpy as np
from activations import sigmoid, relu, softmax

class Dense():
    def __init__(self, ins, outs, activation = None):
        np.random.seed(69)
        self.ins = ins
        self.outs = outs
        self.weights = np.random.uniform(-1, 1, (ins, outs))
        self.bias = np.zeros((1, outs))
        self.activation = activation
        if activation is not None:
            if activation not in ['relu', 'softmax', 'sigmoid']:
                raise RuntimeError("Unknown activation function")
    
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.bias
        if self.activation == 'relu':
            self.outputs = relu(self.outputs)
        elif self.activation == 'sigmoid':
            self.outputs = sigmoid(self.outputs)
        elif self.activation == 'softmax':
            self.outputs - softmax(self.outputs)
    
    def get_output(self):
        return self.outputs