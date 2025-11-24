"""
BERT Fine-tuning Script for Emotion Extremeness Regression

This script fine-tunes a pre-trained BERT model on Best-Worst Scaling (BWS)
labeled data to predict continuous emotion extremeness scores (0.0 to 1.0).

BWS Score Formula: (% Selected Best - % Selected Worst) / Total Appearances

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import os
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmotionDataset(Dataset):
    """Custom Dataset for emotion regression."""

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
            'labels': torch.tensor(label, dtype=torch.float)  # Float for regression
        }


class BERTEmotionRegressor:
    """BERT-based regressor for emotion extremeness using BWS scores."""

    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize the regressor.

        Args:
            model_name: Pre-trained BERT model to use
        """
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None

    def prepare_data(self, df, text_col='text', label_col='extremeness_score'):
        """
        Prepare data for training.

        Args:
            df: DataFrame with text and BWS scores (0.0 to 1.0)
            text_col: Name of text column
            label_col: Name of label column (continuous scores)

        Returns:
            train_loader, val_loader
        """
        texts = df[text_col].values
        labels = df[label_col].values.astype(np.float32)

        # Validate scores are in [0, 1] range
        if labels.min() < 0 or labels.max() > 1:
            print(f"Warning: Scores outside [0,1] range. Min: {labels.min()}, Max: {labels.max()}")
            print("Normalizing to [0, 1]...")
            labels = (labels - labels.min()) / (labels.max() - labels.min())

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
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
        Train the BERT regression model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        # Initialize model with num_labels=1 for REGRESSION
        # This automatically uses MSELoss instead of CrossEntropyLoss
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1  # Single output for regression
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
        best_val_mse = float('inf')

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
                labels = batch['labels'].to(device).unsqueeze(1)  # Shape: (batch, 1)

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
            print(f'Average training loss (MSE): {avg_train_loss:.4f}')

            # Validation
            val_metrics = self.evaluate(val_loader)
            print(f'Validation MSE: {val_metrics["mse"]:.4f}')
            print(f'Validation MAE: {val_metrics["mae"]:.4f}')
            print(f'Validation R²: {val_metrics["r2"]:.4f}')
            print(f'Pearson r: {val_metrics["pearson_r"]:.4f}')

            if val_metrics["mse"] < best_val_mse:
                best_val_mse = val_metrics["mse"]
                print('New best model!')

        print(f'\nBest validation MSE: {best_val_mse:.4f}')
        return best_val_mse

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

                # For regression, logits IS the predicted score
                preds = outputs.logits.squeeze().cpu().numpy()
                if preds.ndim == 0:
                    preds = [preds.item()]
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Clip predictions to [0, 1] range
        predictions = np.clip(predictions, 0, 1)

        # Calculate metrics
        mse = mean_squared_error(true_labels, predictions)
        mae = mean_absolute_error(true_labels, predictions)
        r2 = r2_score(true_labels, predictions)
        pearson_r, _ = pearsonr(true_labels, predictions)

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r
        }

    def predict(self, texts):
        """
        Predict emotion extremeness score for new texts.

        Args:
            texts: Single text string or list of texts

        Returns:
            Continuous scores between 0.0 and 1.0
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

                # The logit IS the extremeness score for regression
                score = outputs.logits.item()
                # Clip to valid range
                score = max(0.0, min(1.0, score))
                predictions.append(round(score, 3))

        return predictions[0] if len(predictions) == 1 else predictions

    def save_model(self, output_dir):
        """Save model, tokenizer, and config to directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        config = {
            'model_name': self.model_name,
            'task': 'regression',
            'num_labels': 1
        }
        with open(os.path.join(output_dir, 'regressor_config.json'), 'w') as f:
            json.dump(config, f)

        print(f'Model saved to {output_dir}')

    def load_model(self, model_dir):
        """Load model from directory."""
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(device)

        print(f'Model loaded from {model_dir}')


def create_example_dataset():
    """
    Create example dataset with BWS-style continuous scores.

    In real BWS, scores are calculated as:
    Score = (% Selected Best - % Selected Worst) / Total Appearances

    This produces values roughly in [-1, 1], normalized to [0, 1].
    """
    np.random.seed(42)

    # Example tweets with BWS-style continuous scores (0.0 = least extreme, 1.0 = most extreme)
    examples = [
        # Low emotion (0.0 - 0.3)
        ("Just published a new report on infrastructure spending.", 0.05),
        ("Meeting with constituents today to discuss local issues.", 0.08),
        ("The committee will review the proposal next week.", 0.10),
        ("Voted on the budget amendment this afternoon.", 0.15),
        ("Pleased to announce new funding for schools.", 0.22),
        ("Looking forward to working across the aisle.", 0.18),
        ("Thank you to everyone who attended the town hall.", 0.12),

        # Medium emotion (0.3 - 0.6)
        ("We must do better for working families!", 0.45),
        ("This is an important step forward for our state.", 0.38),
        ("I strongly support this bipartisan effort.", 0.42),
        ("Our veterans deserve better healthcare options.", 0.48),
        ("Education funding should be a top priority.", 0.40),
        ("We cannot let this opportunity pass us by.", 0.52),
        ("Time to stand up for what's right!", 0.55),

        # High emotion (0.6 - 1.0)
        ("This is a COMPLETE disaster for American workers!", 0.78),
        ("Absolutely unacceptable! We need action NOW!", 0.82),
        ("The other side has FAILED our country!", 0.88),
        ("This is the WORST decision in decades!", 0.91),
        ("OUTRAGEOUS! They are destroying our democracy!", 0.95),
        ("This is an absolute betrayal of the American people!", 0.93),
        ("UNBELIEVABLE! How can they get away with this?!", 0.89),
    ]

    # Expand dataset with variations
    expanded = []
    for text, score in examples:
        expanded.append((text, score))
        # Add lowercase variation with slight score adjustment
        expanded.append((text.lower(), score - 0.02 if score > 0.5 else score))
        # Add exclamation emphasis for high scores
        if score > 0.6:
            expanded.append((text.upper(), min(1.0, score + 0.03)))

    df = pd.DataFrame(expanded, columns=['text', 'extremeness_score'])
    return df


def main():
    parser = argparse.ArgumentParser(description='BERT Emotion Extremeness Regressor')
    parser.add_argument('--train', type=str, help='Path to training CSV file')
    parser.add_argument('--predict', type=str, help='Text to predict')
    parser.add_argument('--model', type=str, default='./emotion_model',
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--text-col', type=str, default='text',
                        help='Name of text column in CSV')
    parser.add_argument('--label-col', type=str, default='extremeness_score',
                        help='Name of BWS score column in CSV (values 0.0-1.0)')
    parser.add_argument('--use-example', action='store_true',
                        help='Use example dataset for demonstration')

    args = parser.parse_args()

    regressor = BERTEmotionRegressor()

    if args.train or args.use_example:
        # Training mode
        if args.use_example:
            print("Using example dataset for demonstration...")
            df = create_example_dataset()
        else:
            print(f"Loading data from {args.train}...")
            df = pd.read_csv(args.train)

        print(f"Dataset size: {len(df)}")
        print(f"Score statistics:")
        print(f"  Mean: {df[args.label_col].mean():.3f}")
        print(f"  Std:  {df[args.label_col].std():.3f}")
        print(f"  Min:  {df[args.label_col].min():.3f}")
        print(f"  Max:  {df[args.label_col].max():.3f}")

        train_loader, val_loader = regressor.prepare_data(
            df, text_col=args.text_col, label_col=args.label_col
        )

        print(f"\nTraining on {device}...")
        regressor.train(train_loader, val_loader, epochs=args.epochs)

        regressor.save_model(args.model)

        # Test predictions
        print("\n" + "=" * 50)
        print("Test Predictions:")
        print("=" * 50)

        test_texts = [
            "This is absolutely OUTRAGEOUS behavior!",
            "I disagree with the vote today.",
            "Just attended a committee meeting."
        ]

        for text in test_texts:
            score = regressor.predict(text)
            print(f"'{text}'")
            print(f"  → Extremeness score: {score}")
            print()

    elif args.predict:
        # Prediction mode
        regressor.load_model(args.model)
        prediction = regressor.predict(args.predict)
        print(f"Emotion extremeness score: {prediction}")

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  Train with example data: python bert_emotion_classifier.py --use-example")
        print("  Train with your data: python bert_emotion_classifier.py --train data.csv")
        print("  Predict: python bert_emotion_classifier.py --predict 'your text' --model ./emotion_model")
        print("\nExpected CSV format:")
        print("  text,extremeness_score")
        print('  "Just posted a report...",0.12')
        print('  "This is OUTRAGEOUS!",0.95')


if __name__ == '__main__':
    main()
