import os

import polars as pl
from pymongo import MongoClient

df = pl.read_csv("data/tsla_tweets.csv")


def add_n_cashtags(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_column(
        pl.col("cashtags")
        .str.replace(r"\[", "")
        .str.replace(r"\]", "")
        .str.split(", ")
        .arr.lengths()
        .alias("n_cashtags")
    )


def remove_links(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_column(
        pl.col("tweet").str.replace_all(r"@\S+|https?://\S+|pic\.twitter\.com/\S+", "")
    )


def n_cashtags_filter(data: pl.DataFrame, n: int = 4) -> pl.DataFrame:
    return data.filter(pl.col("n_cashtags") <= n)


def remove_newlines(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_column(pl.col("tweet").str.replace_all(r"\n", " "))


clean = df.pipe(add_n_cashtags).pipe(remove_links).pipe(n_cashtags_filter)


# print(clean.rename({"tweet": "content"}).select("content").to_dicts())
user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")

client = MongoClient(
    f"mongodb://{user}:{password}@157.90.167.200:27017", authSource="admin"
)
db = client["data_labeling"]
collection = db["dev_coll"]

collection.drop()
collection.insert_many(
    clean.rename({"tweet": "content"})
    .with_columns([pl.lit(None).alias("label"), pl.lit(0).alias("entropy")])
    .select(["content", "label", "entropy"])
    .to_dicts()
)
