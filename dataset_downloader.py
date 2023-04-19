import os


def download_dataset():
    if os.path.exists("data"):
        try:
            import kaggle
        except OSError:
            print("Please authenticate with Kaggle API")
            print("Follow the instructions here:")
            print("https://github.com/Kaggle/kaggle-api#api-credentials")
            exit(1)

        print("Downloading dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests', path='data', unzip=True)
        print("Dataset downloaded successfully!")
