import pandas as pd
import numpy as np
from itertools import combinations
import random

random.seed(114514)

INPUT="data/scraper_result_data/combined/2024/X_2024_combined.csv"
OUTPUT="data/processed/bws_text_data.csv"
NUM_TEXT=3000
GROUP_SIZE=4
APPEARANCE=15

def generate_balanced_design(num_texts, group_size, appearances):

    num_groups = (num_texts * appearances) // group_size
    appearance_count = [0] * num_texts
    # use cooccurrence matrix to store the information as we want as less cooccurence as possible
    cooccurrence = np.zeros((num_texts, num_texts), dtype=int)
    groups = []
    
    for g in range(num_groups):
        group = select_group(
            num_texts, group_size, appearance_count, cooccurrence
        )
        groups.append(group)
        
        for item in group:
            appearance_count[item] += 1
        for i, j in combinations(group, 2):
            cooccurrence[i][j] += 1
            cooccurrence[j][i] += 1

    return groups

def select_group(num_texts, group_size, appearance_count, cooccurrence):
    
    group = []
    
    for _ in range(group_size):
        candidates = [i for i in range(num_texts) if i not in group]
        def score(item):
            app_score = appearance_count[item] * 10000
            coocc_score = 0
            if group:
                coocc_score = sum(cooccurrence[item][g] for g in group)
            # this score filter out text reach appearance count, and punish cooccurance, and add an random shuffle to avoid draw.
            return app_score + (coocc_score * 10) + random.random()
        best_item = min(candidates, key=score)
        group.append(best_item)
    
    return group

df = pd.read_csv(INPUT)
selected_texts = df["content"].sample(n=NUM_TEXT).reset_index(drop=True).tolist()

#use only the hyper-parameter to generate the arrangement
groups = generate_balanced_design(NUM_TEXT, GROUP_SIZE, APPEARANCE)

#make sure the text in each group is in random order
for group in groups:
    random.shuffle(group)

output_data = []
for group_id, group in enumerate(groups):
    row = {"id": group_id}
    for i, text_idx in enumerate(group):
        row[f"text{i+1}"] = selected_texts[text_idx]
    output_data.append(row)

output_df = pd.DataFrame(output_data)
output_df.to_csv(OUTPUT, index=False, encoding="utf-8")