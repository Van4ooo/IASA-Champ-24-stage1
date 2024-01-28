from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow as tf

import numpy as np
import cv2
import pandas as pd
from config import *

CATEGORIES_REV = {
    0: 'alert', 1: 'button', 2: 'card', 3:'checkbox_checked',
    4: 'checkbox_unchecked', 5: 'chip', 6: 'data_table', 7:'dropdown_menu',
    8: 'floating_action_button', 9: 'grid_list', 10: 'image', 11: 'label', 12: 'menu',
    13: 'radio_button_checked', 14: 'radio_button_unchecked', 15: 'slider', 16: 'switch_disabled',
    17: 'switch_enabled', 18: 'text_area', 19: 'text_field', 20: 'tooltip'
}

tf.config.run_functions_eagerly(True)


class InputDate:
    __slots__ = ('images', '__index', 'labels', '_size')
    shape = (224, 224, 1)

    def __init__(self, _size: int) -> None:
        self.images = np.empty((_size,) + self.shape, dtype='float16')
        self.labels = np.empty(_size, dtype='uint8')

        self.__index, self._size = 0, _size

    def add_img(self, _path: str, _categories: int) -> None:
        p = f"data/train/{CATEGORIES_REV[_categories]}/{_path}"
        img = ~cv2.imread(p, 0) / 255

        self.images[self.__index] = np.asarray(img, dtype='float16').reshape(self.shape)
        self.labels[self.__index] = _categories
        self.__index += 1

        if self.__index == self._size:
            self.labels = to_categorical(self.labels)
            print(f"[+] DONE {self._size}/{self._size}")


train_input_f = pd.read_csv('data/processed/train_set.csv')
test_input_f = pd.read_csv('data/processed/test_set.csv')

td = InputDate(len(train_input_f))
test_d = InputDate(len(test_input_f))

for _, path, categories in train_input_f.values:
    td.add_img(path, categories)

for _, path, categories in test_input_f.values:
    test_d.add_img(path, categories)


del train_input_f, test_input_f

train_dataset = tf.data.Dataset.from_tensor_slices((td.images, td.labels))

h_activ, f_activ = 'relu', 'softmax'
kernel_s, pool_s = (3, 3), (2, 2)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_s, activation=h_activ, input_shape=td.shape))
model.add(layers.MaxPooling2D(pool_size=pool_s))
model.add(layers.Conv2D(64, kernel_s, activation=h_activ))
model.add(layers.MaxPooling2D(pool_size=pool_s))
model.add(layers.Conv2D(64, kernel_s, activation=h_activ))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=h_activ))
model.add(layers.Dense(COUNT, activation=f_activ))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=12, batch_size=64, validation_split=0.2)
model.save('models/model_v0.2_wsl.h5')
