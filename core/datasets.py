import numpy as np
import os, random
import cv2

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
    return np.array(gray_img)

def split_data_label(full_data, target_size=None):
    random.shuffle(full_data)
    data = [x for x, _ in full_data]
    label = [y for _, y in full_data]
    # for img, cls in full_data:
    #     data.append(img)
    #     label.append(cls)
    
    width, height = target_size
    data = np.array(data).reshape(-1, height, width)
    label = np.array(label)
    return data, label

def load_from_folder(folder, test_split=0, target_size=None):
    classes = os.listdir(folder)
    train_images = []
    test_images = []
    for cls in classes:
        images = []
        label = classes.index(cls)
        cls_path = os.path.join(folder, cls)
        for img_name in os.listdir(cls_path):
            try:
                img_path = os.path.join(cls_path, img_name)
                img_arr = cv2.imread(img_path)
                if target_size:
                    img_arr = cv2.resize(img_arr, target_size,interpolation=cv2.INTER_CUBIC)
                img_arr = grayscale(img_arr)
                images.append([img_arr, label])
            except Exception as e:
                print(e)
        if test_split: 
            test_size = int(len(images) * (1-test_split))
            train_images += images[:test_size]
            test_images += images[test_size:]
        else :
            train_images += images
    
    train_data, train_label = split_data_label(train_images, target_size)
    # train_data = train_data.reshape(-1, height, width)
    if test_split:
        test_data, test_label = split_data_label(test_images, target_size)
        # test_data = test_data.reshape(-1, height, width)
        return (train_data, train_label), (test_data, test_label)
    return (train_data, train_label)
