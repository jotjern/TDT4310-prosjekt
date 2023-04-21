from tqdm import tqdm


class PredictionModel:
    name = "PredictionModel"

    def train(self, train_df):
        pass

    def predict(self, title: str) -> int:
        pass

    def evaluate(self, test_df):
        correct = 0
        total = 0
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {self.name} model"):
            prediction = self.predict(row["title"])
            if prediction == row["value"]:
                correct += 1
            total += 1

        return correct / total
