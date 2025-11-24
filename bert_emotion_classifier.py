"""
BERT Fine-tuning Script for Emotion Extremeness Classification

This script fine-tunes a pre-trained BERT model on labeled text data
to classify emotion extremeness in social media posts.

Usage:
    python bert_emotion_classifier.py --train data.csv --output ./model
    python bert_emotion_classifier.py --predict "text to classify" --model ./model
"""

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import os
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmotionDataset(Dataset):
    """Custom Dataset for emotion classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTEmotionClassifier:
    """BERT-based classifier for emotion extremeness."""

    def __init__(self, num_labels=5, model_name='bert-base-uncased'):
        """
        Initialize the classifier.

        Args:
            num_labels: Number of emotion extremeness levels (default 5: 1-5 scale)
            model_name: Pre-trained BERT model to use
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_map = None

    def prepare_data(self, df, text_col='text', label_col='emotion_score'):
        """
        Prepare data for training.

        Args:
            df: DataFrame with text and labels
            text_col: Name of text column
            label_col: Name of label column

        Returns:
            train_loader, val_loader
        """
        texts = df[text_col].values
        labels = df[label_col].values

        # Create label mapping if labels are not numeric
        if not np.issubdtype(labels.dtype, np.number):
            unique_labels = sorted(set(labels))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            labels = np.array([self.label_map[l] for l in labels])
            self.num_labels = len(unique_labels)
        else:
            # Assume labels are 1-indexed scores, convert to 0-indexed
            if labels.min() > 0:
                labels = labels - labels.min()
            self.num_labels = len(set(labels))

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = EmotionDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = EmotionDataset(
            val_texts, val_labels, self.tokenizer
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False
        )

        return train_loader, val_loader

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """
        Train the BERT model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(device)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        best_val_accuracy = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 30)

            # Training
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc='Training'):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f'Average training loss: {avg_train_loss:.4f}')

            # Validation
            val_accuracy, val_report = self.evaluate(val_loader)
            print(f'Validation accuracy: {val_accuracy:.4f}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print('New best model!')

        print(f'\nBest validation accuracy: {best_val_accuracy:.4f}')
        return best_val_accuracy

    def evaluate(self, data_loader):
        """Evaluate the model on a dataset."""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)

        return accuracy, report

    def predict(self, texts):
        """
        Predict emotion extremeness for new texts.

        Args:
            texts: Single text string or list of texts

        Returns:
            List of predictions (scores)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")

        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                pred = torch.argmax(outputs.logits, dim=1).item()

                # Convert back to original label if mapping exists
                if self.label_map:
                    reverse_map = {v: k for k, v in self.label_map.items()}
                    pred = reverse_map[pred]
                else:
                    # Add back the offset if labels were 1-indexed
                    pred = pred + 1

                predictions.append(pred)

        return predictions[0] if len(predictions) == 1 else predictions

    def save_model(self, output_dir):
        """Save model, tokenizer, and config to directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save label map and config
        config = {
            'num_labels': self.num_labels,
            'label_map': self.label_map,
            'model_name': self.model_name
        }
        with open(os.path.join(output_dir, 'classifier_config.json'), 'w') as f:
            json.dump(config, f)

        print(f'Model saved to {output_dir}')

    def load_model(self, model_dir):
        """Load model from directory."""
        # Load config
        config_path = os.path.join(model_dir, 'classifier_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.num_labels = config['num_labels']
            self.label_map = config['label_map']
            self.model_name = config['model_name']

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(device)

        print(f'Model loaded from {model_dir}')


def create_example_dataset():
    """Create example dataset for demonstration."""
    np.random.seed(42)

    # Example tweets with varying emotion levels
    examples = [
        # Low emotion (1-2)
        ("Just published a new report on infrastructure spending.", 1),
        ("Meeting with constituents today to discuss local issues.", 1),
        ("The committee will review the proposal next week.", 1),
        ("Voted on the budget amendment this afternoon.", 2),
        ("Pleased to announce new funding for schools.", 2),

        # Medium emotion (3)
        ("We must do better for working families!", 3),
        ("This is an important step forward for our state.", 3),
        ("I strongly support this bipartisan effort.", 3),
        ("Our veterans deserve better healthcare options.", 3),
        ("Education funding should be a top priority.", 3),

        # High emotion (4-5)
        ("This is a COMPLETE disaster for American workers!", 4),
        ("Absolutely unacceptable! We need action NOW!", 4),
        ("The other side has FAILED our country!", 5),
        ("This is the WORST decision in decades!", 5),
        ("OUTRAGEOUS! They are destroying our democracy!", 5),
    ]

    # Expand dataset by slight variations
    expanded = []
    for text, score in examples:
        expanded.append((text, score))
        # Add variations
        expanded.append((text.lower(), score))
        expanded.append((text.upper() if score >= 4 else text, score))

    df = pd.DataFrame(expanded, columns=['text', 'emotion_score'])
    return df


def main():
    parser = argparse.ArgumentParser(description='BERT Emotion Classifier')
    parser.add_argument('--train', type=str, help='Path to training CSV file')
    parser.add_argument('--predict', type=str, help='Text to predict')
    parser.add_argument('--model', type=str, default='./emotion_model',
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--text-col', type=str, default='text',
                        help='Name of text column in CSV')
    parser.add_argument('--label-col', type=str, default='emotion_score',
                        help='Name of label column in CSV')
    parser.add_argument('--use-example', action='store_true',
                        help='Use example dataset for demonstration')

    args = parser.parse_args()

    classifier = BERTEmotionClassifier()

    if args.train or args.use_example:
        # Training mode
        if args.use_example:
            print("Using example dataset for demonstration...")
            df = create_example_dataset()
        else:
            print(f"Loading data from {args.train}...")
            df = pd.read_csv(args.train)

        print(f"Dataset size: {len(df)}")
        print(f"Label distribution:\n{df[args.label_col].value_counts()}")

        train_loader, val_loader = classifier.prepare_data(
            df, text_col=args.text_col, label_col=args.label_col
        )

        print(f"\nTraining on {device}...")
        classifier.train(train_loader, val_loader, epochs=args.epochs)

        classifier.save_model(args.model)

        # Test prediction
        test_text = "This is absolutely OUTRAGEOUS behavior!"
        pred = classifier.predict(test_text)
        print(f"\nTest prediction for '{test_text}': {pred}")

    elif args.predict:
        # Prediction mode
        classifier.load_model(args.model)
        prediction = classifier.predict(args.predict)
        print(f"Emotion extremeness score: {prediction}")

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  Train with example data: python bert_emotion_classifier.py --use-example")
        print("  Train with your data: python bert_emotion_classifier.py --train data.csv")
        print("  Predict: python bert_emotion_classifier.py --predict 'your text' --model ./emotion_model")


if __name__ == '__main__':
    main()
