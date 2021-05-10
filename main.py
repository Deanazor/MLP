# import numpy as np
# from models import MLP
# from optimizers import SGD
# from losses import CategoricalCrossEntropy
# from datasets import create_spiral_data, one_hot_encode
# from datasets import load_from_folder

# np.random.seed(69)

# path = './flowers'

# X,y = create_spiral_data(samples=100, classes=2)
# y = one_hot_encode(y)

# model = MLP(2, 2)

# opt = SGD(lr = 0.1)
# loss = CategoricalCrossEntropy()

# model.compile(opt, loss)

# model.train(X, y, epochs=100)

# train, test = load_from_folder(path, test_split=0.2)
a = list(map(int, input().split()))
x = int(input())

print([a[i:i+x] for i in range(len(a)-x+1)])