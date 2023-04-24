from scipy.sparse import csr_matrix
from tqdm import tqdm
from .prediction_model import PredictionModel
from .sentiment_analysis import sentiment_analysis
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegression(PredictionModel):
    name = "Logistic Regression"
    
    def __init__(self):
        self.count_vectorizer = CountVectorizer(stop_words="english")
        self.model = LR()
        
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]','', text)
        text = re.sub(r'\d+', '', text)
        
        return text
        
    def train(self, train_df):
        train_df = train_df.sample(frac=0.1)
        y = train_df['value']

        titles = []
        sentiment = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Training {self.name} model"):
            i = row['title']
            titles.append(self.preprocess(i))
            sentiment.append(sentiment_analysis(i))
        sentiment = np.array(sentiment).reshape(-1, 1)

        X_titles: csr_matrix = self.count_vectorizer.fit_transform(titles)
        X_titles = X_titles.toarray()

        X = np.concatenate((X_titles, sentiment), axis=1)

        self.model.fit(X, y)
        
        print("Accuracy: ", self.model.score(X, y))
        
    def predict(self, title: str) -> int:
        title_vec = self.count_vectorizer.transform([title])
        title_vec = title_vec.toarray()
        sentiment = sentiment_analysis(title)
        title_vec = np.concatenate((title_vec, np.array(sentiment).reshape(-1, 1)), axis=1)

        return self.model.predict(title_vec)[0]