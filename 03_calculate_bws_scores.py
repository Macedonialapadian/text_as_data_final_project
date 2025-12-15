#(best_count-worst_count)/appearances

import pandas as pd
import numpy as np
from collections import defaultdict
import os

INPUT_FILE = "data/processed/bws_text_data_openai_labelled.csv"
OUTPUT_FILE = "data/processed/bws_scores.csv"

labeled_df = pd.read_csv(INPUT_FILE)
    
text_stats = defaultdict(lambda: {'best_count': 0, 'worst_count': 0, 'appearances': 0})

valid_rows = 0
skipped_rows = 0

for idx, row in labeled_df.iterrows():

    m_val = str(row['most_extreme']).strip()
    l_val = str(row['least_extreme']).strip()

    most_idx = int(float(m_val)) - 1
    least_idx = int(float(l_val)) - 1
    
    valid_rows += 1

    for i, col in enumerate(['text1', 'text2', 'text3', 'text4']):
        text = str(row[col]).strip()
        if text and text.lower() != 'nan':
            text_stats[text]['appearances'] += 1
            
            if i == most_idx:
                text_stats[text]['best_count'] += 1
            if i == least_idx:
                text_stats[text]['worst_count'] += 1

results = []
for text, stats in text_stats.items():
    if stats['appearances'] > 0:
        raw_diff = stats['best_count'] - stats['worst_count']
        
        normalized_score = raw_diff / stats['appearances']
        
        final_score_0_to_1 = (normalized_score + 1) / 2
        
        results.append({
            'text': text,
            'bws_score': final_score_0_to_1,
            'score_original': normalized_score,
            'raw_diff': raw_diff,
            'best_count': stats['best_count'],
            'worst_count': stats['worst_count'],
            'appearances': stats['appearances']
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('bws_score', ascending=False).reset_index(drop=True)
results_df[['text', 'bws_score']].to_csv(OUTPUT_FILE, index=False, encoding='utf-8')