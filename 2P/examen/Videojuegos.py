"""
Problema a resolver:
Una empresa de videojuegos está desarrollando un FPSG (First Person Shooting Game). Para implementar los personajes artificiales del juego, el jefe del proyecto, ex estudiante, ha pensado que podría ser interesante e innovador utilizar una red neuronal. Dicha red tendía que implementar el algoritmo de control de los personajes artificiales, usando los siguientes inputs:
Salud: Débil, Medio y Fuerte
Tiene cuchillo: Si o No
Tiene arma: Si o No
Enemigos: Número de enemigos en el campo visual.
Las acciones que el personaje puede ejecutar son (Ustedes tendrán que generar las reglas de asociación para que el algoritmo identifique cuando podría elegir estas acciones):
Escapar
Andar
Atacar
Esconderse
Resultado:
Esta aplicación de red neuronal deberá de entrenar, predecir y aprender (teniendo en cuenta que el % de selección de cualquiera de las opciones sea superior de 75%).

Nota:
El número de iteraciones será de 120
El número de registros de entrenamiento es de 150
El número de registros para su evaluación es de 5
"""

from pathlib import Path

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

DATA_PATH = Path(__file__).parent / "datos.csv"

dataset = np.loadtxt(DATA_PATH, delimiter=",")

X = dataset[:, :4]
Y = dataset[:, 4]

model = Sequential()
model.add(Dense(12, input_dim=4, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=120, batch_size=10)

predicciones = model.predict(X[:5])

acciones_dict = {
    0: "Esconderse",
    1: "Escapar",
    2: "Andar",
    3: "Atacar",
}

umbral = 0.75
acciones_predichas = []

for i, prediccion in enumerate(predicciones):
    accion_predicha = np.argmax(prediccion)
    if np.max(prediccion) >= umbral:
        acciones_predichas.append((acciones_dict[accion_predicha], acciones_dict[int(Y[i])]))
    else:
        acciones_predichas.append(("No predicho (umbral no alcanzado)", acciones_dict[int(Y[i])]))

for idx, (predicha, real) in enumerate(acciones_predichas, 1):
    print(f"Registro {idx}: Acción predicha: {predicha}, Acción real en el dataset: {real}")
