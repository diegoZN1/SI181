import json
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

BASE_DIR = Path(__file__).parent
INTENTS_PATH = BASE_DIR / "intents.json"
WORDS_PATH = BASE_DIR / "words.pkl"
CLASSES_PATH = BASE_DIR / "classes.pkl"
MODEL_PATH = BASE_DIR / "chatbot_model.h5"

lemmatizer = WordNetLemmatizer()


def train():
    words = []
    classes = []
    documents = []
    ignore_words = ["¿", "?", "!"]

    with open(INTENTS_PATH, encoding="utf-8") as f:
        intents = json.load(f)
    print(intents)

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    print(words)

    with open(CLASSES_PATH, "wb") as f:
        pickle.dump(classes, f)
    with open(WORDS_PATH, "wb") as f:
        pickle.dump(words, f)

    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    train_x = [t[0] for t in training]
    train_y = [t[1] for t in training]

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
    model.save(MODEL_PATH, hist)
    return model


if __name__ == "__main__":
    train()
