import pandas as pd

from models import BagOfWordsModel

train_xs = [
    "I love this product",
    "I hate this product",
    "My wife love this product",
    "My wife hate this product",
]

train_ys = [1, 0, 1, 0]

test_xs = [
    "Who doesn't love this product?",
    "Who doesn't hate this product?",
]

test_ys = [1, 0]

train_df = pd.DataFrame({"title": train_xs, "value": train_ys})
test_df = pd.DataFrame({"title": test_xs, "value": test_ys})

model = BagOfWordsModel()
model.train(train_df)
for x in test_xs:
    print(f"{x} -> {model.predict_float(x)}")
print(f"Model {model.name} accuracy: {model.evaluate(test_df):.2%}")

