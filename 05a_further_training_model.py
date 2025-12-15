import os
import pandas as pd
import torch
import numpy as np
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    LineByLineTextDataset, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)

###############################################################################################
# I did not really ran this script, so I am not sure if it works or if it gives correct results.
###############################################################################################

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest" 
INPUT_CSV = "data/scraper_result_data/combined/2024/X_2024_combined.csv"
OUTPUT_DIR = "models/bws_further_pretrained"
TEMP_TEXT_FILE = "data/processed/corpus_for_pretraining.txt"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MLM_PROBABILITY = 0.15
SEED = 114514

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

set_seed(SEED)

df = pd.read_csv(INPUT_CSV)

texts = df["content"].dropna().astype(str).tolist()

os.makedirs(os.path.dirname(TEMP_TEXT_FILE), exist_ok=True)

with open(TEMP_TEXT_FILE, "w", encoding="utf-8") as f:
    for text in texts:
        clean_text = text.replace("\n", " ").strip()
        if len(clean_text) > 5:
            f.write(clean_text + "\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=TEMP_TEXT_FILE,
    block_size=MAX_LEN,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=MLM_PROBABILITY
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=5000,
    save_total_limit=2,
    learning_rate=LEARNING_RATE,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)