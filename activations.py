import numpy as np

def sigmoid(inputs):
    inputs = np.array(inputs)
    return 1 / (1 + np.exp(-inputs))
    
def relu(inputs):
    return np.maximum(0, inputs)

def softmax(inputs):
    exp_vals = np.exp(inputs - np.max(inputs), axis=1, keepdims=True)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)