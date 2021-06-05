import numpy as np

class sigmoid:
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs
        self.outputs =  1 / (1 + np.exp(-inputs))
    
    def backward(self, dinputs:np.ndarray):
        self.dinputs = dinputs * (1 - self.outputs) * self.outputs

class relu:    
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs
        self.outputs = np.maximum(0.1, inputs)
    
    def backward(self, dinputs:np.ndarray):
        self.dinputs = dinputs.copy()
        self.dinputs[self.inputs <= 0] = 0

class softmax:
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    def backward(self, dinputs:np.ndarray):
        self.dinputs = np.empty_like(dinputs)
        for i, (output, dinput) in enumerate(zip(self.outputs, dinputs)):
            output = output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, dinput)
