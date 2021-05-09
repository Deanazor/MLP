import numpy as np

class CategoricalCrossEntropy:
    def forward(self, preds:np.ndarray, actual:np.ndarray):
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        samples = len(preds)
        if len(actual.shape) == 1:
            confidence = preds_clipped[range(samples),actual]
        
        elif len(actual.shape) == 2:
            confidence = np.sum(preds_clipped * actual, axis=1)

        return -np.log(confidence)
    
    def calc_loss(self, preds:np.ndarray, actual:np.ndarray):
        losses = self.forward(preds, actual)
        return np.mean(losses)

    def backward(self, dinputs:np.ndarray, actual:np.ndarray):
        samples = len(dinputs)
        labels = len(dinputs[0])

        if len(actual.shape) == 1:
            actual = np.eye(labels)[actual]
        
        self.dinputs = -actual/dinputs
        self.dinputs = self.dinputs / samples