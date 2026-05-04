from pathlib import Path

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

DATA_PATH = Path(__file__).parent / "pima-indians-diabetes.csv"

np.random.seed(7)

dataset = np.loadtxt(DATA_PATH, delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=150, batch_size=10)

predictions = model.predict(X)
print(predictions)

rounded = [round(x[0]) for x in predictions]
print(rounded)
