from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential, load_model
from keras.layers import Dense

# newModel = load_model("../models/mysupermodel.h5")

data = genfromtxt("../data/bank_note_data.txt", delimiter=",")

labels = data[:, 4]

features = data[:, :4]

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler_object = MinMaxScaler()

# TODO : Bak
scaler_object.fit(X_train)

# Formülü
# (x - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

model = Sequential()

model.add(Dense(4, input_dim=4, activation="relu"))

model.add(Dense(8, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(scaled_X_train, y_train, epochs=50, verbose=2)

predictions = model.predict_classes(scaled_X_test)

x = confusion_matrix(y_test, predictions)

model.save("mysupermodel.h5")
q = classification_report(y_test, predictions)

print(q)
