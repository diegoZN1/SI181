# SI181 — Sistemas Inteligentes

Programas de la materia **Sistemas Inteligentes** (8vo cuatrimestre, Ingeniería en Sistemas Computacionales, Universidad Politécnica de Querétaro).

Repositorio refactorizado a estructura production-grade: rutas relativas con `pathlib`, dependencias declaradas, y chatbot modularizado para futura exposición como API.

---

## Estructura

```
SI181/
├── requirements.txt
├── 2P/                              # Segundo Parcial — MLPs con Keras
│   ├── diabetes/
│   │   ├── diabetes_eval.py         # Entrena + evalúa accuracy
│   │   ├── diabetes_predict.py      # Entrena + predice
│   │   └── pima-indians-diabetes.csv
│   ├── examen/
│   │   ├── Videojuegos.py           # FPS NPC: 4 acciones, softmax
│   │   └── datos.csv
│   └── financiera/
│       ├── Financiera.py            # Clasificación binaria fiabilidad cliente
│       ├── datos_clientes.csv
│       └── pred.csv
└── 3P/                              # Tercer Parcial — NLP + Chatbot
    ├── sentimientos/
    │   ├── sentimientos.py          # TextBlob + VADER + SentiWordNet
    │   └── comentarios.csv
    ├── AnálisisSentimientosGrupal/
    │   ├── grupo_sentimientos.py    # Análisis sobre Google Forms
    │   └── forms181.xlsx
    └── chatbot/
        ├── train.py                 # Entrena modelo (genera .h5 + .pkl)
        ├── chat.py                  # Inferencia: get_chatbot_response() + CLI
        ├── chatUI.py                # GUI Tkinter
        ├── intents.json             # Patrones + respuestas
        ├── chatbot_model.h5         # Modelo entrenado
        ├── words.pkl
        └── classes.pkl
```

---

## Setup

Python 3.10+ recomendado.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

Recursos NLTK (primera vez):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('sentiwordnet')
nltk.download('omw-1.4')
```

---

## Uso

### 2P — Modelos Keras

```bash
python 2P/diabetes/diabetes_eval.py
python 2P/diabetes/diabetes_predict.py
python 2P/examen/Videojuegos.py
python 2P/financiera/Financiera.py
```

### 3P — Análisis de Sentimientos

```bash
python 3P/sentimientos/sentimientos.py
python 3P/AnálisisSentimientosGrupal/grupo_sentimientos.py
```

Genera gráficas pie-chart con matplotlib (TextBlob, VADER, SentiWordNet).

### 3P — Chatbot

**1. Entrenar modelo** (solo si no existe `chatbot_model.h5`):

```bash
cd 3P/chatbot
python train.py
```

**2. CLI:**

```bash
python chat.py
# You: hola
# Bot: ...
# You: salir
```

**3. GUI Tkinter:**

```bash
python chatUI.py
```

**4. Uso programático (futuro API):**

```python
from chat import get_chatbot_response

reply = get_chatbot_response("¿Qué horarios manejan?")
```

---

## Convenciones

- **Rutas:** todas relativas al archivo vía `Path(__file__).parent / "data.csv"`. No depende del CWD.
- **Imports:** estándar primero, terceros después, locales al final.
- **Chatbot:** `chat.py` carga modelo + intents al importarse (load-once para servir requests). Para FastAPI/Flask, importar y exponer `get_chatbot_response`.

---

## Roadmap a API

1. Envolver `get_chatbot_response` en endpoint FastAPI:
   ```python
   from fastapi import FastAPI
   from chat import get_chatbot_response

   app = FastAPI()

   @app.post("/chat")
   def chat(message: str):
       return {"reply": get_chatbot_response(message)}
   ```
2. Empaquetar `chatbot/` con `__init__.py` para imports limpios.
3. Dockerizar (TF + modelo `.h5` en imagen).
4. Cache LRU de respuestas frecuentes.

---

## Stack

| Dominio | Librerías |
|---------|-----------|
| Deep Learning | TensorFlow / Keras |
| ML clásico | scikit-learn |
| NLP | NLTK, TextBlob, VADER, SentiWordNet |
| Datos | pandas, numpy, openpyxl |
| Traducción | deep-translator (Google) |
| Visualización | matplotlib |
| GUI | Tkinter (stdlib) |

---

## Autor

Diego Zamora — UPQ, 8vo cuatrimestre ISC.
