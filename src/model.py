import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from config import *

CATEGORIES_REVERSE = {v: k for k, v in CATEGORIES_CODES.items()}


class DataLoader:
    __slots__ = ('images', '__index', 'labels', '_size', 'data')
    shape = (224, 224, 1)

    def __init__(self, _path: str) -> None:
        self.data = pd.read_csv(_path)
        self._size = self.data.shape[0]
        self.images = np.empty((self._size,) + self.shape, dtype='float16')
        self.labels = np.empty(self._size, dtype='uint8')
        self.__index = 0

    def load_data(self) -> None:
        for _, _path, _categories in self.data.values:
            self.add_image(_path, _categories)

    def add_image(self, _path: str, category: int) -> None:
        image_path = f"data/train/{CATEGORIES_REVERSE[category]}/{_path}"
        img = ~cv2.imread(image_path, 0) / 255
        self.images[self.__index] = np.asarray(img, dtype='float16').reshape(self.shape)
        self.labels[self.__index] = category
        self.__index += 1
        if self.__index == self._size:
            self.labels = to_categorical(self.labels)
            print(f"[+] DONE {self._size}/{self._size}")
            return
        print(f"[+] Progress: {self.__index}/{self._size}")

    def show_image(self, index: int) -> None:
        plt.imshow(self.images[index], cmap='gray')
        plt.show()
        plt.close()


class SketchClassificatorModel:
    __slots__ = ('_data', '_shape', 'num_classes', 'model', 'version')

    def __init__(self, data, _version: str) -> None:
        self._data = data
        self._shape = data.shape
        self.num_classes = len(CATEGORIES_CODES)
        self.model = models.Sequential()
        self.version = _version

    def build_model(self) -> None:
        activation, final_activation = 'relu', 'softmax'
        kernel_size, pool_size = (3, 3), (2, 2)

        self.model.add(layers.Conv2D(32, kernel_size, activation=activation, input_shape=self._shape))
        self.model.add(layers.MaxPooling2D(pool_size=pool_size))
        self.model.add(layers.Conv2D(64, kernel_size, activation=activation))
        self.model.add(layers.MaxPooling2D(pool_size=pool_size))
        self.model.add(layers.Conv2D(64, kernel_size, activation=activation))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation=activation))
        self.model.add(layers.Dense(self.num_classes, activation=final_activation))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self) -> None:
        self.model.fit(self._data.images, self._data.labels, epochs=9, batch_size=64, validation_split=0.2)

    def save(self) -> None:
        self.model.save(f'models/model_v{self.version}.h5')


class PredictWithModel:
    shape = (224, 224, 1)

    def __init__(self, version: str) -> None:
        self.test_img_path = 'data/test'
        self.model = load_model(f'models/model_v{version}.h5')
        self.path = os.listdir(self.test_img_path)
        self.images = np.empty((_s := len(self.path),) + self.shape, dtype='float16')
        self.predictions = pd.DataFrame(columns=['ID', 'CATEGORY'])

    def parse_img(self, img_path: str) -> np.array:
        img = ~cv2.imread(f"{self.test_img_path}/{img_path}", 0) / 255
        return np.asarray(img, dtype='float16').reshape(self.shape)

    def predict_img(self):
        for i, img_path in enumerate(self.path):
            self.images[i] = self.parse_img(img_path)

        for i, p in enumerate(self.model.predict(self.images)):
            self.predictions.loc[i] = [self.path[i], CATEGORIES_REVERSE[np.argmax(p)]]
        self.predictions.to_csv('data/result/predictions.csv')

    @staticmethod
    def show(img: np.array) -> None:
        plt.imshow(img[0], cmap='gray')
