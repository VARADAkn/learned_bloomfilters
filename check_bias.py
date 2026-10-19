import pandas as pd

df = pd.read_csv("data/malicious_phish.csv").dropna(subset=["url"]).drop_duplicates("url")
df["url"] = df["url"].astype(str)
df["safe"] = df["type"] == "benign"
bare = df["url"].str.replace(r"^(?:https?://)?(?:www\.)?", "", regex=True)
df["has_path"] = bare.str.contains("/")
df["length"] = bare.str.len()
print(df.groupby("safe")[["has_path", "length"]].mean().round(3))
print(df.groupby("type")["has_path"].mean().round(3))