from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix

from .prediction_model import PredictionModel

from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class NaïveBayes(PredictionModel):
    name = "Naïve Bayes"

    def __init__(self):
        self.model = MultinomialNB()
        self.count_vectorizer = CountVectorizer(stop_words="english")
        self.sia = SentimentIntensityAnalyzer()

    def getSentiment(self, text):
        return self.sia.polarity_scores(text)['compound']

    def train(self, train_df):
        train_df = train_df.sample(frac=0.1)
        y = train_df['value']

        titles = []
        sentiment = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Training {self.name} model"):
            i = row['title']
            titles.append(self.preprocess(i))
            sentiment.append((self.getSentiment(i) + 1) / 2)  # map sentiment value to range [0, 1]
        sentiment = np.array(sentiment).reshape(-1, 1)

        X_titles: csr_matrix = self.count_vectorizer.fit_transform(titles)
        X_titles = X_titles.toarray()

        X = np.concatenate((X_titles, sentiment), axis=1)

        self.model.fit(X, y)
        
        print("Accuracy: ", self.model.score(X, y))

    def predict(self, title: str) -> int:
        title_vec = self.count_vectorizer.transform([title])
        title_vec = title_vec.toarray()
        sentiment = self.getSentiment(title)
        title_vec = np.concatenate((title_vec, np.array(sentiment).reshape(-1, 1)), axis=1)

        return self.model.predict(title_vec)[0]