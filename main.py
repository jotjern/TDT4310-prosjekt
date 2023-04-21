import pandas as pd

import dataset_downloader
from tqdm import tqdm

from models import BagOfWordsModel, SentimentAnalysisModel, PredictionModel


def main():
    print("Loading data...")
    train_df, test_df = dataset_downloader.load_data()

    models = [BagOfWordsModel(), SentimentAnalysisModel()]

    for model in models:
        model.train(train_df)
        print(f"Model {model.name} accuracy: {model.evaluate(test_df):.2%}")


if __name__ == "__main__":
    main()
