from typing import List, Iterable

from sklearn.pipeline import Pipeline

from .prediction_model import PredictionModel

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

class NaïveBayes(PredictionModel):
    name = "Naïve Bayes"

    def __init__(self, model=MultinomialNB):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.model = model()
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])

    def train(self, train_df):
        self.pipeline.fit(train_df['title'], train_df['value'])

    def predict_float(self, title: str) -> float:
        return self.predict(title)

    def predict(self, title: str) -> int:
        return self.pipeline.predict([title])[0]

    def predict_all(self, titles: List[str]) -> Iterable[int]:
        return self.pipeline.predict(titles)
