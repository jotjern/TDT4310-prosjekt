from typing import Callable

import pandas as pd

import dataset_downloader
from pyarrow import feather
from tqdm import tqdm
from collections import Counter
from sentiment_analysis import sentiment_analysis


def f1_evaluate_model(test_df: pd.DataFrame, predict: Callable[[str], int], model_name: str = "") -> float:
    correct = 0
    total = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating model {model_name}"):
        prediction = predict(row["title"])
        if prediction == row["value"]:
            correct += 1
        total += 1

    return correct / total


def main():
    dataset_downloader.download_kaggle_dataset()
    dataset_downloader.download_yahoo_stock_data()

    df = feather.read_feather("data/training_data.feather")
    # drop empty title ""
    df = df[(df["title"] != "") & (df["title"] != " ")]
    train_df = df[df["is_train"]]
    test_df = df[~df["is_train"]]

    word_data = {}
    for _, row in train_df.iterrows():
        for word in row["title"].lower().split():
            if word not in word_data:
                word_data[word] = {"positive": 0, "negative": 0}

            if row["value"] == 1:
                word_data[word]["positive"] += 1
            else:
                word_data[word]["negative"] += 1

    word_weights = {}
    for word, weights in word_data.items():
        positive = weights["positive"]
        negative = weights["negative"]
        total = positive + negative
        if total > 0:
            word_weights[word] = positive / total

    def predict_bag_of_words(title: str) -> int:
        return sum(
            word_weights.get(word) for word in title.lower().split() if word in word_weights) / len(
            title.split()
        ) > 0.5

    def predict_sentiment_analysis(title: str) -> int:
        return sentiment_analysis(title) > 0

    bag_of_words_f1 = f1_evaluate_model(test_df, predict_bag_of_words, "Bag of words")
    sentiment_analysis_f1 = f1_evaluate_model(test_df, predict_sentiment_analysis, "Sentiment analysis")

    print("Bag of words F1:", bag_of_words_f1)
    print("Sentiment analysis F1:", sentiment_analysis_f1)


if __name__ == "__main__":
    main()
