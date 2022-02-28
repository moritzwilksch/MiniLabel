import os

import polars as pl
from bson.objectid import ObjectId
from pymongo import MongoClient
from rich.console import Console


class MongoConnector:
    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        db: str,
        collection: str,
        authSource: str = "admin",
    ) -> None:
        self.client = MongoClient(
            f"mongodb://{user}:{password}@{host}:{port}/{db}", authSource=authSource
        )
        self.db = self.client[db]
        self.collection = self.db[collection]

    def import_data(self, data: pl.DataFrame) -> None:
        """
        Imports data into the database.

        Args:
            data: polars data frame with columns "content" and "label"
        """
        if set(data.columns) != {"content", "label"}:
            raise ValueError("Columns of data frame are not (just): content, label.")

        c = Console()
        confirm_prompt = c.input(
            f"[red][ATTENTION][/] This overwrites the entire collection. Are you sure you want to continue? [yes/no]: "
        )

        if confirm_prompt != "yes":
            c.print("[green][ABORT][/] Data import aborted.")
            return

        # convert "None" to actual null value
        data = data.with_column(
            pl.when(pl.col("label") == "None")
            .then(None)
            .otherwise(pl.col("label"))
            .alias("label")
        )

        self.collection.drop()
        self.collection.insert_many(data.to_dicts())
        c.print("[green][DONE][/] Data imported successfully.")

    def load_all_data(self) -> pl.DataFrame:
        """
        Loads all data from DB as polars.DataFrame.

        Returns:
            Data frame with columns "_id", "content", "label"
        """
        res = list(self.collection.find({}, {"_id": 1, "content": 1, "label": 1}))
        for rec in res:
            rec["_id"] = str(rec["_id"])

        return pl.from_dicts(res)

    def update_one(self, id: str, label: str) -> None:
        """
        Update label of one single document with id.

        Args:
            id: MongoDB ID of the document as str
            label: Label to set
        """

        self.collection.update_one({"_id": ObjectId(id)}, {"$set": {"label": label}})


if __name__ == "__main__":
    user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    conn = MongoConnector(
        user,
        password,
        host="157.90.167.200",
        port=27017,
        db="data_labeling",
        collection="dev_coll",
    )

    # df = pl.DataFrame({"content": ["a", "b", "casdf"], "label": ["pos", "neg", "None"]})
    # conn.import_data(df)

    data = conn.load_all_data()
    print(data)
