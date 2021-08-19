import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from PIL import Image

# Load the model
model = load_model('keras_model.h5')


def fonk(img):
    image = Image.fromarray(np.uint8(img)).convert('RGB')

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array
    return data


# run the inference
prediction = model.predict(fonk(Image.open('../data/Nadia_Murad.jpg')))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame = None
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(100)

    data = fonk(frame)
    p = model.predict_classes(data)
    print(p)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
