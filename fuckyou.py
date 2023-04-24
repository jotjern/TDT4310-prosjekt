import numpy as np

X_titles = np.load("X_titles.npy", allow_pickle=True)
sentiment = np.load("sentiment.npy", allow_pickle=True)

print(X_titles.dtype, X_titles.shape)
print(sentiment.dtype, sentiment.shape)
