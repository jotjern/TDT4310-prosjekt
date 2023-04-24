from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()

def analyse_sentiment(df):
    positive = 0
    negative = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if sia.polarity_scores(row["title"])['compound'] > 0:
            positive += 1
        else:
            negative += 1

    print(f"Positive: {positive}")
    print(f"Negative: {negative}")
    
def analyse_value(df):
    positive = 0
    negative = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['value'] == 1:
            positive += 1
        else:
            negative += 1
            
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")