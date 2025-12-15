import pandas as pd

df = pd.read_csv("data/processed/final_bws_dataset.csv", dtype={'url': str})
df_full = pd.read_csv(
    "data/scraper_material_data/x_2024.csv", dtype={"urls": str, "official_id": str}
)

df_urls_official_id = pd.DataFrame(columns=["url", "official_id"])

df_full["urls"] = df_full["urls"].astype(str).str.split(", ")
df_urls_official_id = df_full.explode("urls").rename(columns={"urls": "url"})
df_urls_official_id = df_urls_official_id.reset_index(drop=True)
df_urls_official_id["url"] = df_urls_official_id["url"].astype(str).str.strip()


df = pd.merge(df, df_urls_official_id, on="url", how="left")

df["comments"] = df["comments"].fillna(0)
df["retweets"] = df["retweets"].fillna(0)
df["likes"] = df["likes"].fillna(0)
df = df.drop(columns=["quotes"])
df["views"] = df["views"].fillna(0)

df.to_csv("data/cleaned/the_dataset.csv", index=False)
df_urls_official_id.to_csv("data/cleaned/urls_official_id.csv", index=False)
