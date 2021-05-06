import numpy as np
from activations import sigmoid, relu, softmax

class Dense():
    def __init__(self, ins, outs, activation = None):
        np.random.seed(69)
        self.ins = ins
        self.outs = outs
        self.weights = np.random.uniform(-1, 1, (ins, outs))
        self.bias = np.zeros((1, outs))
    
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.bias
    
    def get_output(self):
        return self.outputs