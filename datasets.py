import numpy as np
import os
from PIL import Image
import cv2

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

def grayscale(img):
    bgr_weights = np.array([.1140, .5870, .2989])
    gray_img = np.dot(img, bgr_weights)
    return gray_img

def load_from_folder(folder, test_split=0, target_size=None):
    classes = os.listdir(folder)
    train_images = []
    test_images = []
    for cls in classes:
        images = []
        label = classes.index(cls)
        cls_path = os.path.join(folder, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, target_size,interpolation=cv2.INTER_CUBIC)
            img_arr = grayscale(img_arr)
            images.append([img_arr, label])
        if test_split: 
            test_size = int(len(images) * (1-test_split))
            train_images += images[:test_size]
            test_images += images[test_size:]
        else :
            train_images += images
    if test_split:
        return train_images, test_images
    return train_images
