import numpy as np
from models import MLP
from datasets import create_spiral_data, one_hot_encode

np.random.seed(69)

X,y = create_spiral_data(samples=100, classes=2)
y = one_hot_encode(y)

model = MLP(2, 2, lr=0.1)

model.train(X, y, epochs=100)