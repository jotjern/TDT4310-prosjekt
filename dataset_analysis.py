import pandas as pd
from tqdm import tqdm
from pyarrow import feather
from nltk import word_tokenize
from multiprocessing.pool import Pool

def preprocess_title(title):
    return word_tokenize(title.lower())

def main():
    train_df = feather.read_feather("data/training_data.feather")
    train_df = train_df[train_df["is_train"]]

    pool = Pool(8)

    train_df["words"] = list(tqdm(pool.imap(
        preprocess_title, train_df["title"]), total=len(train_df), desc="Tokenizing titles"))

    train_df["sentiment"] = "positive"
    train_df.loc[train_df["value"] == 0, "sentiment"] = "negative"

    # Explode the DataFrame on the "words" column
    exploded_df = train_df.explode("words")

    # Create a crosstab to count positive and negative occurrences for each word
    word_counts = pd.crosstab(exploded_df['words'], exploded_df['sentiment'])

    # Convert the crosstab to a dictionary format
    word_data = word_counts.to_dict('index')

    word_scores = []

    for word, weights in word_data.items():
        positive = weights["positive"]
        negative = weights["negative"]
        total = positive + negative

        if total > 100:
            word_scores.append((word, positive / total))

    word_scores.sort(key=lambda x: x[1], reverse=True)

    print("Most positive words:")
    for word, weight in word_scores[:50]:
        print(f"{word}: {weight:.2%}")
    word_scores.sort(key=lambda x: x[1])
    print("Most negative words:")
    for word, weight in word_scores[:50]:
        print(f"{word}: {weight:.2%}")


if __name__ == "__main__":
    main()