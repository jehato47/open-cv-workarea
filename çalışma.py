from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4, 4), activation="relu", input_shape=(32, 32, 3)))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(4, 4), activation="relu", input_shape=(32, 32, 3)))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, "relu"))

model.add(Dense(10, "softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit(x_train, y_cat_train, verbose=1, epochs=2)

model.evaluate(x_test, y_cat_test)

predictions = model.predict_classes(x_test)


