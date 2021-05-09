import numpy as np

class SGD:
    def __init__(self, lr=1e-3, momentum=0):
        self.learning_rate = lr
        self.momentum = momentum
    
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.bias)
            
            weight_updates = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.learning_rate * layer.dbias
            layer.bias_momentums = bias_updates
        
        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbias
        
        layer.weights += weight_updates
        layer.bias += bias_updates