import re
from typing import List, Iterable

import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from multiprocessing.pool import Pool

ENGLISH_STOPWORDS = set(stopwords.words('english'))


class PredictionModel:
    name = "PredictionModel"
    lemmatizer = WordNetLemmatizer()
    
    def train(self, train_df):
        pass

    def predict_float(self, title: str) -> float:
        pass

    def predict(self, title: str) -> int:
        return self.predict_float(title) > 0.5

    def predict_all(self, titles: List[str]) -> Iterable[int]:
        return (self.predict(title) for title in titles)

    def preprocess(self, text) -> List[str]:
        text = text.lower()

        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        words = word_tokenize(text)
        words = (self.lemmatizer.lemmatize(word) for word in words)
        words = (word for word in words if word not in ENGLISH_STOPWORDS)
        return list(words)

    def _evaluate_batch(self, test_df):
        predictions = self.predict_all(test_df['title'])
        tp, tn, fp, fn = 0, 0, 0, 0
        for prediction, value in zip(predictions, test_df['value']):
            if prediction == value:
                if value == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if value == 1:
                    fn += 1
                else:
                    fp += 1
        return tp, tn, fp, fn

    def evaluate(self, test_df, n_batches=100, n_threads=1):
        iterator = np.array_split(test_df, n_batches)
        if n_threads > 1:
            pool = Pool(n_threads)
            iterator = pool.imap_unordered(self._evaluate_batch, iterator)
        else:
            iterator = map(self._evaluate_batch, iterator)

        tp, tn, fp, fn = 0, 0, 0, 0
        for chunk in tqdm(iterator, total=n_batches, desc=f"Evaluating {self.name} model"):
            tn += chunk[0]
            tp += chunk[1]
            fn += chunk[2]
            fp += chunk[3]

        return tp, tn, fp, fn
