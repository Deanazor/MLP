import numpy as np
from models import MLP
from optimizers import SGD
from losses import CategoricalCrossEntropy
from datasets import one_hot_encode
from datasets import load_from_folder
from sklearn.datasets import make_classification

np.random.seed(69)

path = './flowers'

# X, y = make_classification(n_samples=10000, n_classes=3, n_features=100, n_redundant=2, n_clusters_per_class=1)
# y = one_hot_encode(y)
# print(X.shape)

width = 240
height = 320

(train_data, train_label), (test_data, test_label) = load_from_folder(path, test_split=0.2, target_size=(width, height))

train_data = train_data.reshape(-1, height*width)
test_data = test_data.reshape(-1, height*width)

train_label = one_hot_encode(train_label)
test_label = one_hot_encode(test_label)

model = MLP(height*width, 3)

opt = SGD(lr = 0.1, momentum=0.1)
loss = CategoricalCrossEntropy()

# model.compile(opt, loss)

model.train(train_data, train_label, epochs=100)