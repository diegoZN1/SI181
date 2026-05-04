import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from deep_translator import GoogleTranslator
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_PATH = Path(__file__).parent / "comentarios.csv"

# Paso1: Limpieza y traducción de texto
mydata = pd.read_csv(DATA_PATH, delimiter=",")
mydata.head()
print(mydata)

translator = GoogleTranslator(source="es", target="en")
mydata.review = mydata.review.apply(translator.translate)
print(mydata)


def clean(text):
    text = re.sub("[^A-Za-z]+", " ", text)
    return text


mydata["Cleaned_Reviews"] = mydata["review"].apply(clean)
mydata.head()
print(mydata)

# Paso 2: Tokenización + POS + stopwords
pos_dict = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}


def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words("english")):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


mydata["POS_tagged"] = mydata["Cleaned_Reviews"].apply(token_stop_pos)
mydata.head()
print(mydata)

# Paso 3: Lematización
wordnet_lemmatizer = WordNetLemmatizer()


def lematize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


mydata["Lemma"] = mydata["POS_tagged"].apply(lematize)
mydata.head()
print(mydata[["review", "Lemma"]])


# Paso 4: análisis con TextBlob
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


fin_data = pd.DataFrame(mydata[["review", "Lemma"]])
fin_data["Subjetividad"] = fin_data["Lemma"].apply(getSubjectivity)
fin_data["Polaridad"] = fin_data["Lemma"].apply(getPolarity)
fin_data["Resultado"] = fin_data["Polaridad"].apply(analysis)
fin_data.head()
print(fin_data)

tb_counts = fin_data["Resultado"].value_counts()
print(tb_counts)

plt.figure(figsize=(10, 7))
plt.title("Resultado TextBlob")
plt.pie(tb_counts.values, labels=tb_counts.index, explode=(0.1, 0, 0), autopct="%1.1f%%", shadow=False)
plt.show()

# VADER
analyzer = SentimentIntensityAnalyzer()


def vadersentimentanalysis(comentario):
    vs = analyzer.polarity_scores(comentario)
    return vs["compound"]


fin_data["Vader_Sentiment"] = fin_data["Lemma"].apply(vadersentimentanalysis)


def vader_analysis(compound):
    if compound >= 0.5:
        return "Positivo"
    elif compound <= -0.5:
        return "Neutro"
    else:
        return "Negativo"


fin_data["Vader_Analysis"] = fin_data["Vader_Sentiment"].apply(vader_analysis)
fin_data.head()
print(fin_data)

vader_counts = fin_data["Vader_Analysis"].value_counts()
print(vader_counts)

plt.figure(figsize=(10, 7))
plt.title("Resultado Vader_Analysis")
plt.pie(vader_counts.values, labels=vader_counts.index, explode=(0.1, 0, 0), autopct="%1.1f%%", shadow=False)
plt.show()


# SentiWordNet
def sentiwordnetanalysis(pos_data):
    sentiment = 0
    tokens_count = 0
    for word, pos in pos_data:
        if not pos:
            continue
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue
        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1
        print(swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score())
    if not tokens_count:
        return 0
    if sentiment > 0:
        return "Positivo"
    if sentiment == 0:
        return "Neutral"
    else:
        return "Negativo"


fin_data["SWN_Analysis"] = mydata["POS_tagged"].apply(sentiwordnetanalysis)
print(fin_data["SWN_Analysis"])
fin_data.head()
print(fin_data)


swn_counts = fin_data["SWN_Analysis"].value_counts()
print(swn_counts)

explode = [0] * len(swn_counts.index)
explode[swn_counts.index.get_loc("Positivo")] = 0.1

plt.figure(figsize=(10, 7))
plt.title("Resultados SentiWordNet")
plt.pie(swn_counts.values, labels=swn_counts.index, explode=explode, autopct="%1.1f%%", shadow=False)
plt.show()
