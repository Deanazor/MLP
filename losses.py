import numpy as np

class CategoricalCrossEntropy():
    def forward(self, preds, actual):
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        confidence = np.sum(preds_clipped*actual, axis=1)
        return -np.log(confidence)
    
    def calc_loss(self, preds, actual):
        losses = self.forward(preds, actual)
        return np.mean(losses)

    def backward(self, dinputs, actual):
        samples = len(dinputs)
        labels = len(dinputs[0])

        if len(actual.shape == 1):
            actual = np.eye(labels)[actual]
        
        self.dinputs = -actual/dinputs
        self.dinputs = self.dinputs / samples