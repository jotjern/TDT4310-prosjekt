from tqdm import tqdm
from .prediction_model import PredictionModel
from .sentiment_analysis import sentiment_analysis
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
        y = train_df['value']
        
        titles = []
        sentiment = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Training {self.name} model"):
            i = row['title']
            titles.append(self.preprocess(i))
            sentiment.append(sentiment_analysis(i))
        
        X_titles = self.count_vectorizer.fit_transform(titles)
        X = list(zip(X_titles.toarray(), [s for _, s in sentiment]))
        print(X)
        
        self.model.fit(X, y)
        
        print("Accuracy: ", self.model.score(X, y))
        
    def predict(self, title: str) -> int:
        title_vec = self.count_vectorizer.transform([title])
        
        return self.model.predict(title_vec)[0]