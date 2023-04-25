from typing import List, Iterable

from sklearn.pipeline import Pipeline
from .prediction_model import PredictionModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel


class LogisticRegression(PredictionModel):
    name = "Logistic Regression"
    
    def __init__(self):
        self.count_vectorizer = TfidfVectorizer(stop_words="english")
        self.logistic_regression = LogisticRegressionModel(C=0.1, solver="liblinear", penalty="l2", max_iter=10000000)
        self.model = Pipeline([
            ('vectorizer', self.count_vectorizer),
            ('classifier', self.logistic_regression)
        ])

    def train(self, train_df):
        self.model.fit(train_df['title'], train_df['value'])

    def predict(self, title: str) -> int:
        return self.model.predict([title])[0]

    def predict_all(self, titles: List[str]) -> Iterable[int]:
        return self.model.predict(titles)
