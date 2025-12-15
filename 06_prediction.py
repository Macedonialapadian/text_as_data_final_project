import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
import os
import numpy as np

MODEL_DIR = "models/bws_regressor_final"
INPUT_CSV = "data/scraper_result_data/combined/2024/X_2024_combined.csv"
OUTPUT_CSV = "data/processed/final_bws_dataset.csv"
BATCH_SIZE = 32
MAX_LEN = 128

device = torch.device('cuda')
    
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

df = pd.read_csv(INPUT_CSV)

texts = df["content"].astype(str).tolist()

input_ids = []
attention_masks = []

for sent in tqdm(texts, desc="Tokenizing"):
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=BATCH_SIZE)

predictions = []

for batch in tqdm(dataloader, desc="Inference"):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    
    with torch.no_grad():
        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = result.logits
    predictions.extend(logits.cpu().numpy().flatten())

df['predicted_bws_score'] = predictions

df.to_csv(OUTPUT_CSV, index=False)
