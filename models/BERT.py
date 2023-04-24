from .prediction_model import PredictionModel

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup

class BertPredictionModel(PredictionModel):
    def __init__(self, num_labels=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )

    def train(self, train_df, epochs=4, batch_size=16, learning_rate=2e-5):
        # Tokenize the input data
        X_train = self.tokenizer.batch_encode_plus(
            train_df['title'].values,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=64,
            return_tensors='pt'
        )
        y_train = torch.tensor(train_df['value'].values)

        # Create dataloader
        train_data = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Train the model
        self.model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            for step, batch in enumerate(train_dataloader):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

        print("Training complete!")

    def predict(self, title: str) -> int:
        self.model.eval()
        encoded_title = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=64,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded_title['input_ids'].to(self.device)
        attention_mask = encoded_title['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

        return predictions.item()
