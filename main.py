import numpy as np
from models import MLP
from optimizers import SGD
from losses import CategoricalCrossEntropy
from datasets import create_spiral_data, one_hot_encode

np.random.seed(69)

X,y = create_spiral_data(samples=100, classes=2)
y = one_hot_encode(y)

model = MLP(2, 2)

opt = SGD(lr = 0.1)
loss = CategoricalCrossEntropy()

model.compile(opt, loss)

model.train(X, y, epochs=100)