import dataset_downloader
from pyarrow import feather
from tqdm import tqdm
from collections import Counter
from sentiment_analysis import sentiment_analysis

if __name__ == "__main__":
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

    n_correct = 0
    for _, row in tqdm(test_df.iterrows(), desc="Testing bag of words model"):
        predicted_value = sum(
            word_weights.get(word) for word in row["title"].lower().split() if word in word_weights) / len(
            row["title"].split()
        ) > 0.5

        if predicted_value == row["value"]:
            n_correct += 1

    print(f"Bag of words F1 score: {n_correct / len(test_df)}")

    n_correct = 0
    for _, row in tqdm(test_df.iterrows(), desc="Testing sentiment analysis model"):
        predicted_value = sentiment_analysis(row["title"]) > 0

        if predicted_value == row["value"]:
            n_correct += 1

    print(f"Sentiment analysis F1 score: {n_correct / len(test_df)}")
