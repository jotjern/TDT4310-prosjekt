from datetime import timedelta
from pyarrow import feather
import pandas as pd
import yahoo_fin.stock_info
import numpy as np
import os


def download_kaggle_dataset():
    # Downoad the dataset from Kaggle if it doesn't exist
    if not os.path.exists("data"):
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

    # Cache the dataset as a feather file if it doesn't exist for faster loading
    if not os.path.exists("data/analyst_ratings_processed.feather"):
        print("Converting CSV to Feather format...")

        import pandas as pd
        from pyarrow import feather

        df = pd.read_csv("data/analyst_ratings_processed.csv")
        df = df.dropna(subset=["stock", "date"])
        df = df.drop_duplicates(subset=["title"])
        df["date"] = pd.to_datetime(df["date"])
        # Convert to naive datetime and remove time from date
        df["date"] = df["date"].apply(lambda x: x.replace(tzinfo=None).date())
        df["is_train"] = np.random.choice([True, False], size=len(df), p=[0.8, 0.2])

        feather.write_feather(df, "data/analyst_ratings_processed.feather")
        print("Conversion complete!")


def download_yahoo_stock_data():
    if os.path.exists("data/training_data.feather"):
        return

    df = feather.read_feather("data/analyst_ratings_processed.feather")
    df["date"] = df["date"].astype("datetime64[ns]")
    stock_dfs = {}
    for stock, stock_df in df.groupby("stock"):
        lowest_date, highest_date = stock_df["date"].agg(["min", "max"])
        # Add a day to the start and end dates to make sure we get all the data
        lowest_date = lowest_date - timedelta(days=1)
        highest_date = highest_date + timedelta(days=1)

        cache_fname = os.path.join("yahoo_fin_cache", f"{stock}_{lowest_date}_{highest_date}.feather")

        if not os.path.exists(cache_fname):
            try:
                data = yahoo_fin.stock_info.get_data(stock, start_date=lowest_date, end_date=highest_date)
                print("Downloaded data for", stock, cache_fname)
            except (AssertionError, KeyError):
                stock_df = pd.DataFrame()
            else:
                data.index.name = "date"

                stock_df = df[df["stock"] == stock].merge(data[["open", "close"]], on="date")

            feather.write_feather(stock_df, cache_fname)
        else:
            stock_df = feather.read_feather(cache_fname)

        stock_dfs[stock] = stock_df

    df = pd.concat(stock_dfs.values())
    df["delta"] = df["close"] - df["open"]
    df["value"] = df["delta"].apply(lambda x: 1 if x > 0 else 0)

    feather.write_feather(df, "data/training_data.feather")

