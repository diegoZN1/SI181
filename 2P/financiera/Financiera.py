from pathlib import Path

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "datos_clientes.csv"
PRED_PATH = BASE_DIR / "pred.csv"

datos_entrenamiento = np.genfromtxt(TRAIN_PATH, delimiter=",", dtype=np.float32)

X_train = datos_entrenamiento[:, 2:]
y_train = (datos_entrenamiento[:, 2] >= 25) & (datos_entrenamiento[:, 3] >= 29000)
y_train = y_train.astype(int)

scaler_edad = MinMaxScaler()
scaler_ingresos = MinMaxScaler()

X_train[:, 0:1] = scaler_edad.fit_transform(X_train[:, 0:1])
X_train[:, 1:2] = scaler_ingresos.fit_transform(X_train[:, 1:2])

model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

datos_prediccion = np.genfromtxt(PRED_PATH, delimiter=",", dtype=np.float32)

X_pred = datos_prediccion[:, 2:]

X_pred[:, 0:1] = scaler_edad.transform(X_pred[:, 0:1])
X_pred[:, 1:2] = scaler_ingresos.transform(X_pred[:, 1:2])

predictions = model.predict(X_pred)

y_pred = np.round(predictions).astype(int)

print("Predicciones:")
print(y_pred)
