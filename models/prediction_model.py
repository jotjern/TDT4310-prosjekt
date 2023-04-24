import re
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class PredictionModel:
    name = "PredictionModel"
    lemmatizer = WordNetLemmatizer()
    
    def train(self, train_df):
        pass

    def predict(self, title: str) -> int:
        pass

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]','', text)
        text = re.sub(r'\d+', '', text)
        words = word_tokenize(text)

        words = [self.lemmatizer.lemmatize(word) for word in words if word not in stopwords.words()]
        text = " ".join(words)
        return text

    def evaluate(self, test_df):
        correct = 0
        total = 0
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {self.name} model"):
            prediction = self.predict(row["title"])
            if prediction == row["value"]:
                correct += 1
            total += 1

        return correct / total
