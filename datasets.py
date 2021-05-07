import numpy as np

def create_spiral_data(samples = 10, classes=2, dimension=2):
    X = np.zeros((samples*classes, dimension))
    y = np.zeros(samples*classes, dtype='uint8')

    for i in range(classes):
        ix = range(samples*i, samples*(i+1))
        r = np.linspace(0,1,samples)
        theta = np.linspace(i*4, (i+1)*4, samples) + np.random.rand(samples) * 0.2
        X[ix] = np.c_[r*np.cos(theta), r*np.sin(theta)]
        y[ix] = i
    
    return X, y

def one_hot_encode(y):
    samples = len(y)
    n_classes = len(np.unique(y))
    categorical = np.zeros((samples,n_classes))
    for i, label in enumerate(y):
        categorical[i, label] = 1
    return categorical