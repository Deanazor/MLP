import numpy as np
from activations import sigmoid, relu, softmax

class Dense():
    def __init__(self, ins, outs):
        self.ins = ins
        self.outs = outs
        self.weights = np.random.uniform(-1, 1, (ins, outs))
        self.bias = np.zeros((1, outs))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias
    
    def backward(self, dinputs):
        self.dweights = np.dot(self.inputs.T, dinputs)
        self.dbias = np.sum(dinputs, axis=0, keepdims=True)
        self.dinputs =  np.dot(dinputs, self.weights.T)