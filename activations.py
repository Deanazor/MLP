import numpy as np

class sigmoid():
    def forward(self, inputs):
        inputs = np.array(inputs)
        self.outputs =  1 / (1 + np.exp(-inputs))

class relu():    
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

class softmax():
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs), axis=1, keepdims=True)
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)