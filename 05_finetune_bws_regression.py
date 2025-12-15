import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import time
import datetime
import random
import os

DATA_PATH = "data/processed/bws_final_dataset.csv"
OUTPUT_DIR = "models/bws_regressor_final"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
K_FOLDS = 5
SEED = 114514

def good_update_interval(total_iters, num_desired_updates):
    if total_iters == 0: return 1
    exact_interval = total_iters / num_desired_updates
    order_of_mag = len(str(total_iters)) - 1
    if order_of_mag < 1: return 1
    round_mag = order_of_mag - 1
    update_interval = int(round(exact_interval, -round_mag))
    if update_interval == 0:
        update_interval = 1
    return update_interval

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def compute_metrics(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(labels, preds)
    spearman_corr, _ = spearmanr(labels, preds)
    return {"rmse": rmse, "pearson": pearson_corr, "spearman": spearman_corr}

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input_ids = []
attention_masks = []

for sent in df['text']:
    encoded_dict = tokenizer.encode_plus(
        str(sent),
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
labels = torch.tensor(df['bws_score'].values, dtype=torch.float)
dataset = TensorDataset(input_ids, attention_masks, labels)

kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    val_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch_i in range(0, EPOCHS):
        model.train()
        t0 = time.time()
        total_train_loss = 0
        
        update_interval = good_update_interval(len(train_dataloader), 5)
        
        for step, batch in enumerate(train_dataloader):
            if step % update_interval == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = result.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds = []
        val_labels = []
        
        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():
                result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            logits = result.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            val_preds.extend(logits)
            val_labels.extend(label_ids)

        metrics = compute_metrics(np.array(val_preds), np.array(val_labels))
        
    fold_results.append(metrics['pearson'])

    del model
    torch.cuda.empty_cache()


full_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(dataset))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=1,
    ignore_mismatched_sizes=True 
)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(full_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch_i in range(0, EPOCHS):
    model.train()
    for step, batch in enumerate(full_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = result.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)