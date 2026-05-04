import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from deep_translator import GoogleTranslator
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

EXCEL_PATH = Path(__file__).parent / "forms181.xlsx"

mydata = pd.read_excel(EXCEL_PATH)


def translate_text(text):
    return GoogleTranslator(source="es", target="en").translate(text)


mydata["pr1"] = mydata["pr1"].apply(translate_text)
mydata["pr2"] = mydata["pr2"].apply(translate_text)
print(mydata)


def clean(text):
    text = re.sub("[^A-Za-z]+", " ", text)
    return text


mydata["pr1"] = mydata["pr1"].apply(clean)
mydata["pr2"] = mydata["pr2"].apply(clean)
print(mydata)

pos_dict = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}


def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words("english")):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


wordnet_lemmatizer = WordNetLemmatizer()


def lematize(pos_data):
    lemma_rew = ""
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


mydata["POS_tagged_pr1"] = mydata["pr1"].apply(token_stop_pos)
mydata["POS_tagged_pr2"] = mydata["pr2"].apply(token_stop_pos)

mydata["Lemma_pr1"] = mydata["POS_tagged_pr1"].apply(lematize)
mydata["Lemma_pr2"] = mydata["POS_tagged_pr2"].apply(lematize)

print(mydata[["pr1", "Lemma_pr1"]])
print(mydata[["pr2", "Lemma_pr2"]])


def getSubjectivity(comentarios):
    return TextBlob(comentarios).sentiment.subjectivity


def getPolarity(comentarios):
    return TextBlob(comentarios).sentiment.polarity


def analysis(score):
    if score < 0:
        return "Negativo"
    elif score == 0:
        return "Neutro"
    else:
        return "Positivo"


fin_data_pr1 = pd.DataFrame(mydata[["pr1", "Lemma_pr1"]])
fin_data_pr2 = pd.DataFrame(mydata[["pr2", "Lemma_pr2"]])

fin_data_pr1["Subjetividad"] = fin_data_pr1["Lemma_pr1"].apply(getSubjectivity)
fin_data_pr1["Polaridad"] = fin_data_pr1["Lemma_pr1"].apply(getPolarity)
fin_data_pr1["Resultado"] = fin_data_pr1["Polaridad"].apply(analysis)

fin_data_pr2["Subjetividad"] = fin_data_pr2["Lemma_pr2"].apply(getSubjectivity)
fin_data_pr2["Polaridad"] = fin_data_pr2["Lemma_pr2"].apply(getPolarity)
fin_data_pr2["Resultado"] = fin_data_pr2["Polaridad"].apply(analysis)

print("Resultado para Describe tu experiencia en las clases de los martes y jueves este cuatrimestre:")
print(fin_data_pr1)
print("\nResultado para ¿Qué emociones experimentas al pensar en la clase del profesor Nelson?:")
print(fin_data_pr2)

print("Gráfica de los resultados para Describe tu experiencia en las clases de los martes y jueves este cuatrimestre:")
tb_counts_pr1 = fin_data_pr1["Resultado"].value_counts()
print(tb_counts_pr1)

plt.figure(figsize=(10, 7))
plt.title("TextBlob Describe tu experiencia en las clases de los martes y jueves este cuatrimestre")
plt.pie(tb_counts_pr1.values, labels=tb_counts_pr1.index, explode=(0.1, 0, 0), autopct="%1.1f%%", shadow=False)
plt.show()

print("Gráfica de los resultados para ¿Qué emociones experimentas al pensar en la clase del profesor Nelson?:")
tb_counts_pr2 = fin_data_pr2["Resultado"].value_counts()
print(tb_counts_pr2)

plt.figure(figsize=(10, 7))
plt.title("TextBlob ¿Qué emociones experimentas al pensar en la clase del profesor Nelson?")
plt.pie(tb_counts_pr2.values, labels=tb_counts_pr2.index, explode=(0.1, 0, 0), autopct="%1.1f%%", shadow=False)
plt.show()
