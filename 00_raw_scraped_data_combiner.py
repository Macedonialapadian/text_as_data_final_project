import pandas as pd
import os
import glob

RAW_SCRAPED_DATA_PATHS = "data/scraper_result_data/raw/2024"
COMBINED_DATA_PATH = "data/scraper_result_data/combined/2024"


if not os.path.exists(COMBINED_DATA_PATH):
    os.makedirs(COMBINED_DATA_PATH)

dta_all=pd.DataFrame()
dta=pd.DataFrame()

csv_files = glob.glob(os.path.join(RAW_SCRAPED_DATA_PATHS, "*.csv"))

for csv_file in csv_files:
    dta = pd.read_csv(csv_file, dtype=str,engine='python')
    dta_all = pd.concat([dta_all, dta], ignore_index=True)

dta_all = dta_all.drop_duplicates(subset=['url'], keep='first')
dta_all = dta_all.sort_values('datetime', ascending=True)

dta_all.to_csv(os.path.join(COMBINED_DATA_PATH, "X_2024_combined.csv"), index=False, encoding='utf-8')

