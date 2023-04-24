import sys

import pandas as pd

import dataset_downloader
from tqdm import tqdm

from models import BagOfWordsModel, SentimentAnalysisModel, PredictionModel, LogisticRegression, NaïveBayes, BertPredictionModel
import data_analysis


def main():
    print("Loading data...")
    train_df, test_df = dataset_downloader.load_data()
    train_df = train_df.sample(frac=0.01)

    models = [LogisticRegression()]

    for model in models:
        model.train(train_df)

        f1_evaluation = model.evaluate(test_df)

        print()
        print(f"Model {model.name} accuracy: {f1_evaluation:.2%}")


if __name__ == "__main__":
    main()
