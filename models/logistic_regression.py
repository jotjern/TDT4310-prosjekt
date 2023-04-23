from .prediction_model import PredictionModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegression(PredictionModel):
    name = "Logistic Regression"
    
    def __init__(self):
        self.count_vectorizer = CountVectorizer(stop_words="english")
        self.model = LR()
        
    def train(self, train_df):
        x = self.count_vectorizer.fit_transform(train_df['title'])
        y = train_df['value']
        
        self.model.fit(x, y)
        
        print("Accuracy: ", self.model.score(x, y))
        
    def predict(self, title: str) -> int:
        title_vec = self.count_vectorizer.transform([title])
        
        return self.model.predict(title_vec)[0]