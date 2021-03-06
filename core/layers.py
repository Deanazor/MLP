import numpy as np
from .activations import softmax
from .losses import CategoricalCrossEntropy

class Dense:
    def __init__(self, ins:int, outs:int):
        self.ins = ins
        self.outs = outs
        self.weights = np.random.randn(ins, outs) * 0.01
        self.bias = np.zeros((1, outs))
    
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias
    
    def backward(self, dinputs:np.ndarray):
        self.dweights = np.dot(self.inputs.T, dinputs)
        self.dbias = np.sum(dinputs, axis=0, keepdims=True)
        self.dinputs =  np.dot(dinputs, self.weights.T)

class Output:
    def __init__(self):
        self.activation = softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs:np.ndarray, actual:np.ndarray = None):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        if actual is not None:
            return self.loss.calc_loss(self.outputs, actual)
    
    def backward(self, dinputs:np.ndarray, actual:np.ndarray):
        samples = len(dinputs)

        if len(actual.shape) == 2:
            actual = np.argmax(actual, axis=1)
        
        self.dinputs = dinputs.copy()
        self.dinputs[range(samples), actual] -= 1
        self.dinputs = self.dinputs / samples
