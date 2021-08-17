import warnings
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential, load_model
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image

warnings.filterwarnings("ignore")
# from keras.preprocessing.image import ImageDataGenerator

plt.interactive(True)

image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1 / 255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode="nearest",
                               )

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

train_image_gen = image_gen.flow_from_directory("../data/CATS_DOGS/train",
                                                target_size=(150, 150),
                                                batch_size=16,
                                                class_mode="binary",
                                                )

test_image_gen = image_gen.flow_from_directory("../data/CATS_DOGS/test",
                                               target_size=(150, 150),
                                               batch_size=16,
                                               class_mode="binary",
                                               )

# results = model.fit_generator(train_image_gen, epochs=5, steps_per_epoch=150, validation_data=test_image_gen, validation_steps=12)

new_model = load_model("cat_dog_100epochs.h5")

dog_img = image.load_img("../data/CATS_DOGS/test/DOG/10005.jpg", target_size=(150, 150))

dog_img = image.img_to_array(dog_img)

dog_img = np.expand_dims(dog_img, axis=0)

dog_img = dog_img / 255

print(new_model.predict_classes(dog_img))
print(new_model.predict(dog_img))
