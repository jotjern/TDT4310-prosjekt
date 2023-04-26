from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

import random
import nltk

from .prediction_model import PredictionModel


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return ADJ
    elif tag.startswith('V'):
        return VERB
    elif tag.startswith('R'):
        return ADV
    else:
        return NOUN


def sentiment_analysis(sentence):
    lemmatizer = WordNetLemmatizer()
    sentiment = 0.0
    tokens_count = 0

    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)

    for word, tag in tagged_words:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        synsets = wn.synsets(lemma, pos=wn_tag)

        if not synsets:
            continue

        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        word_sentiment = swn_synset.pos_score() - swn_synset.neg_score()

        sentiment += word_sentiment
        tokens_count += 1

    if tokens_count == 0:
        return 0

    sentiment_score = sentiment / tokens_count
    return sentiment_score

class SentimentAnalysisModel(PredictionModel):
    name = "Sentiment analysis"

    def predict(self, title: str) -> int:
        sentiment = sentiment_analysis(title)
        if sentiment > 0:
            return 1
        elif sentiment < 0:
            return 0
        else:
            return random.Random(hash(title)).randint(0, 1)

    
