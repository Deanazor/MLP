import numpy as np
from .layers import Dense
from .activations import relu, sigmoid
from .metric import accuracy

class Model:    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.compiled = True
    

class MLP(Model):
    def __init__(self, input_shape, output_shape,):
        self.dense1 = Dense(input_shape, 4)
        self.activation1 = relu()
        self.dense2 = Dense(4, output_shape)
        self.activation2 = sigmoid()
        self.compiled = False

    def forward(self, X):
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.outputs)

        self.dense2.forward(self.activation1.outputs)
        self.activation2.forward(self.dense2.outputs)
    
    def backward(self, y):
        self.loss.backward(self.activation2.outputs, y)
        self.activation2.backward(self.loss.dinputs)
        self.dense2.backward(self.activation2.dinputs)

        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)
    
    def optimize(self):
        self.optimizer.update_params(self.dense1)
        self.optimizer.update_params(self.dense2)
    
    def train(self, X, y, epochs=10,validation_data=None, return_logs=False):
        if not self.compiled:
            raise RuntimeError("Model needs to be compiled")

        logs = {"loss" : [],
                "acc" : [],
                "val_loss" : [],
                "val_acc" : []}
        
        if validation_data:
            X_val, y_val = validation_data

        for i in range(epochs):
            # Forward Propagation / Feed Forward
            self.forward(X)
            loss = self.loss.calc_loss(self.activation2.outputs, y)

            preds = np.argmax(self.activation2.outputs, axis=1)
            acc = accuracy(y, preds)
            
            print("epoch {}/{}: loss: {}; acc: {};".format(i+1, epochs, loss, acc), end=" ")

            # Back Propagation / Backward pass
            self.backward(y)
            self.optimize()

            if validation_data:
                val_pred, val_loss = self.validate(X_val, y_val)
                val_acc = accuracy(y_val, val_pred)
                print("val_loss: {}; val_acc: {};".format(val_loss, val_acc), end=" ")

            if return_logs:
                logs['loss'].append(loss)
                logs['acc'].append(acc)
                if validation_data:
                    logs['val_acc'].append(val_acc)
                    logs['val_loss'].append(val_loss)
            print()
            
        if return_logs:
            return logs

    def validate(self, X, y):
        self.forward(X)
        loss = self.loss.calc_loss(self.activation2.outputs, y)
        preds = np.argmax(self.activation2.outputs, axis=1)
        return preds, loss

    def predict(self, X):
        self.forward(X)
        prediction = np.argmax(self.activation2.outputs, axis=1)
        return prediction
    