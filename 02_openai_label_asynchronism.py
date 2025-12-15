import os
import re
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = "data/processed/bws_text_data.csv"
OUTPUT_PATH = "data/processed/bws_text_data_openai_labelled.csv"
SAVE_INTERVAL = 500
MAX_CONCURRENT = 15
TIMEOUT_SECONDS = 60

client = AsyncOpenAI(
    base_url="http://api.yesapikey.com/v1"
)

async def chat_with_retry(semaphore, prompt: str, system_prompt: str, retries=3) -> str:
    async with semaphore:
        for attempt in range(retries):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="gpt-5-mini-2025-08-07",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        #my third party API have conditions using these parameters.
                        #max_tokens=5,
                        #temperature=0.0,
                    ),
                    timeout=TIMEOUT_SECONDS
                )
                return response.choices[0].message.content

            except asyncio.TimeoutError:
                if attempt < retries - 1:
                    print(f"Retrying {attempt+1}")
                    await asyncio.sleep(2) 
                else:
                    print(f"Timeout error after {retries} attempts.")
            except Exception as e:
                print(e)

async def process_batch(df, batch_indices, semaphore, system_prompt):
    tasks = []
    valid_indices = []

    # create request
    for i in batch_indices:
        current_most = str(df.at[i, "most_extreme"]).strip()
        if current_most and current_most.lower() != "nan":
            continue
            
        row = df.iloc[i]
        prompt = f'1. "{row["text1"]}"\n2. "{row["text2"]}"\n3. "{row["text3"]}"\n4. "{row["text4"]}"'
        
        task = chat_with_retry(semaphore, prompt, system_prompt)
        tasks.append(task)
        valid_indices.append(i)

    if not tasks:
        return False

    #send the request
    results = await asyncio.gather(*tasks)

    #deal with the result
    updates = 0
    for idx, response in zip(valid_indices, results):
        if response:
            match = re.search(r'(\d+)\s*,\s*(\d+)', response)
            most = match.group(1)
            least = match.group(2)
            if most and least:
                df.at[idx, "most_extreme"] = most
                df.at[idx, "least_extreme"] = least
                print(f"Row {idx} Success: {most}, {least}")
                updates += 1
            else:
                print(f"Row {idx} Parse Error: {raw_response}")
        else:
            print(f"Row {idx} Failed")
    
    return updates > 0

async def main():
    df = pd.read_csv(INPUT_PATH)

    # read the previous work so that we can continue on it.
    df_existing = pd.read_csv(OUTPUT_PATH)
    if "most_extreme" in df_existing.columns:
        df["most_extreme"] = df_existing["most_extreme"].fillna("")
        df["least_extreme"] = df_existing["least_extreme"].fillna("")
    
    if "most_extreme" not in df.columns:
        df["most_extreme"] = ""
        df["least_extreme"] = ""

    system_prompt = """We need to generate best-worst scaling scores.
    Output ONLY two numbers separated by a comma (most,least).
    Example: 1,3"""

    #this parameter is to set the max concurrence 
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    total_rows = len(df)
    for i in range(0, total_rows, SAVE_INTERVAL):
        batch_end = min(i + SAVE_INTERVAL, total_rows)
        batch_indices = range(i, batch_end)
        
        needs_save = await process_batch(df, batch_indices, semaphore, system_prompt)
        
        if needs_save:
            df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(main())