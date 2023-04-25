import dataset_downloader

from models import BagOfWordsModel, SentimentAnalysisModel, LogisticRegression, NaïveBayes


def get_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def get_f1_score(tp, _tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


def main():
    print("Loading data...")
    train_df, test_df = dataset_downloader.load_data()

    models = [LogisticRegression(), NaïveBayes(), BagOfWordsModel(), SentimentAnalysisModel()]

    for model in models:
        print(f"Training {model.name} model...")
        model.train(train_df)

        tn, tp, fn, fp = model.evaluate(test_df, n_threads=8, n_batches=1000)
        f1_accuracy = get_f1_score(tp, tn, fp, fn)
        accuracy = get_accuracy(tp, tn, fp, fn)

        print(f"Model {model.name} accuracy: {accuracy:.2%} F1: {f1_accuracy:.2%}")


if __name__ == "__main__":
    main()
