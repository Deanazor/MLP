from layers import Dense
from activations import relu
import numpy as np

np.random.seed(69)

X = [[1.5, 2, 3.7],
     [4.3, 5, 2.1],
     [8, 3.2, 5.6]]

layer = Dense(3, 3)
activation = relu()
layer.forward(X)
activation.forward(layer.outputs)
print(activation.outputs)

