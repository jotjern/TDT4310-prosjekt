from tqdm import tqdm

from .prediction_model import PredictionModel


class BagOfWordsModel(PredictionModel):
    name = "Bag of words"

    def __init__(self):
        self.word_weights = {}

    def train(self, train_df):
        word_data = {}
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Training {self.name} model"):
            for word in row["title"].lower().split():
                if word not in word_data:
                    word_data[word] = {"positive": 0, "negative": 0}

                if row["value"] == 1:
                    word_data[word]["positive"] += 1
                else:
                    word_data[word]["negative"] += 1

        for word, weights in word_data.items():
            positive = weights["positive"]
            negative = weights["negative"]
            total = positive + negative
            if total > 0:
                self.word_weights[word] = positive / total

    def predict_float(self, title: str) -> float:
        weights = [self.word_weights.get(word) for word in title.lower().split() if word in self.word_weights]
        return sum(weights) / len(weights)

    def predict(self, title: str) -> int:
        return self.predict_float(title) > 0.5
