import os
import numpy as np
import cv2
from keras.utils import to_categorical
class DataLoader:
    def __init__(self):
        self.x_train = []
        self.y_train = []

    def load_data(self, path = 'data/'):
        LABELS = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']
        amount = 7
        datasets = os.listdir(path)
        data = []
        y = []
        for dataset_path in datasets:
            images = os.listdir(path+dataset_path)
            for image in images:
                i = cv2.imread(os.path.join(path + dataset_path,image))
                i = cv2.resize(i,(32,32))
                i = i/255.0
                data.append(i)
                y.append(LABELS.index(dataset_path))
        self.x_train, self.y_train = np.array(data), np.array(y)
        self.y_train = to_categorical(self.y_train, amount)


        