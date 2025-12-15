import pandas as pd

INPUT_FILE = "data/processed/bws_scores.csv"
OUTPUT_FILE = "data/processed/bws_final_dataset.csv"

df = pd.read_csv(INPUT_FILE)
    
df = df.dropna(subset=['text', 'bws_score'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 5]
df['bws_score'] = pd.to_numeric(df['bws_score'], errors='coerce')
df = df.dropna(subset=['bws_score'])
    
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

