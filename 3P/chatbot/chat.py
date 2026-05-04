import json
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "chatbot_model.h5"
INTENTS_PATH = BASE_DIR / "intents.json"
WORDS_PATH = BASE_DIR / "words.pkl"
CLASSES_PATH = BASE_DIR / "classes.pkl"

ERROR_THRESHOLD = 0.25

lemmatizer = WordNetLemmatizer()

_model = keras.models.load_model(MODEL_PATH)
with open(INTENTS_PATH, encoding="utf-8") as _f:
    _intents = json.load(_f)
with open(WORDS_PATH, "rb") as _f:
    _words = pickle.load(_f)
with open(CLASSES_PATH, "rb") as _f:
    _classes = pickle.load(_f)


def _clean_up_sentence(sentence: str) -> list[str]:
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in sentence_words]


def _bow(sentence: str, words: list[str]) -> np.ndarray:
    sentence_words = _clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def _predict_class(sentence: str) -> list[dict]:
    p = _bow(sentence, _words)
    res = _model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": _classes[r[0]], "probability": str(r[1])} for r in results]


def _get_response(ints: list[dict], intents_json: dict) -> str:
    if not ints:
        return "Lo siento, no entendí tu mensaje."
    tag = ints[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Lo siento, no entendí tu mensaje."


def get_chatbot_response(user_input: str) -> str:
    """Return chatbot reply for given user input. API-ready entry point."""
    ints = _predict_class(user_input)
    return _get_response(ints, _intents)


if __name__ == "__main__":
    print("Chatbot listo. Escribe 'salir' para terminar.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            break
        print(f"Bot: {get_chatbot_response(user_input)}")
