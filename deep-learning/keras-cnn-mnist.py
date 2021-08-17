import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

plt.interactive(True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

single_image = x_train[0]

# plt.imshow(single_image, cmap="gray_r")

y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(4, 4),
                 input_shape=x_train[0].shape,
                 activation="relu",
                 ), )

model.add(MaxPool2D(pool_size=(2, 2)))
# 2d -> 1d
model.add(Flatten())

model.add(Dense(128, "relu"))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
#
# model.fit(x_train, y_cat_train, epochs=2)
#
# model.evaluate(x_test, y_cat_test)
#
predictions = model.predict_classes(x_test)
#
print(classification_report(y_test, predictions))
