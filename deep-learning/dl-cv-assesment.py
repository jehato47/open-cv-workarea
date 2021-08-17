import numpy as np
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train: np.ndarray
x_test: np.ndarray

x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=x_train[0].shape, activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit(x_train, y_cat_train, epochs=10)

predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))
