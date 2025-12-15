import pandas as pd
import os
import glob
import re
"""
my first scraper did not handle csvs with commas, so that if the "content" contains commas then the csv rows are broken.
But I know that the first 2 and last 5 elements are fixed format.
So I manually find the these elements to get the position of the "content".
there is also line breaks in the "content", so i add buffer to store incomplete lines.
If the script is not able to find the last 5 elements, then it means the line is broken.
Then it will attacth the next line to the buffer, until it finds the last 5 elements.
"""
RAW_SCRAPED_DATA_PATHS = "data/scraper_result_data/raw/2024"
COMBINED_DATA_PATH = "data/scraper_result_data/combined/2024/X_2024_combined.csv"

def parse_broken_csv(file_path):
    data = []
    buffer = ""
    tail_pattern = re.compile(r',((?:[\d\.]+|NA|nan)),((?:[\d\.]+|NA|nan)),((?:[\d\.]+|NA|nan)),((?:[\d\.]+|NA|nan)),((?:[\d\.]+|NA|nan))\s*$')

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n')

                if line.startswith("url,datetime") or line.startswith('"url","datetime"') or line.startswith("url,created_at"):
                    continue

                if buffer:
                    current_line = buffer + " " + line
                else:
                    current_line = line
                
                match = tail_pattern.search(current_line)
                if match:
                    likes, retweets, comments, quotes, views = match.groups()
                    head_and_content = current_line[:match.start()]
                    parts = head_and_content.split(',', 2)
                    
                    if len(parts) == 3:
                        url_raw, date_raw, content_raw = parts
                        url = url_raw.strip('"').strip()
                        datetime = date_raw.strip('"').strip()
                        content = content_raw.strip()
                        if content.startswith('"') and content.endswith('"'):
                            content = content[1:-1]
                        content = content.replace('""', '"')
                        data.append({
                            'url': url,
                            'datetime': datetime,
                            'content': content,
                            'likes': likes,
                            'retweets': retweets,
                            'comments': comments,
                            'quotes': quotes,
                            'views': views
                        })
                    buffer = ""
                else:
                    buffer = current_line
                    
    except Exception as e:
        print(e)

    return pd.DataFrame(data)

dta_all = pd.DataFrame()

csv_files = glob.glob(os.path.join(RAW_SCRAPED_DATA_PATHS, "*.csv"))

for csv_file in csv_files:
    dta = parse_broken_csv(csv_file)
    dta_all = pd.concat([dta_all, dta], ignore_index=True)

dta_all = dta_all.drop_duplicates(subset=['url'], keep='first')
dta_all = dta_all.sort_values('datetime', ascending=True)

dta_all.to_csv(COMBINED_DATA_PATH, index=False, encoding='utf-8')
