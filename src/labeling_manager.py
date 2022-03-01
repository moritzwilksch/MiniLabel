import json

import polars as pl
from bson.objectid import ObjectId

from src.database.connectors import MongoConnector
from src.ml_models.base_model import MLModel


class LabelingManager:
    def __init__(self, db_connector: MongoConnector, model: MLModel) -> None:
        self.db_connector = db_connector
        self.model = MLModel

        self._load_data_from_db()
        self._current_get_sample_retries = 0

    def _load_data_from_db(self):
        self.data = self.db_connector.load_all_data()

    def _add_sampling_weight(self):
        ...

    def _retrain_model(self):
        ...

    def get_sample(self, max_retries=1) -> dict:
        if self._current_get_sample_retries > max_retries:
            # TODO: graceful handling of this
            raise RuntimeError("No more data to sample.")

        try:
            json_string = (
                self.data.filter(pl.col("label").is_null())
                .sample(n=1)
                .to_json(to_string=True, json_lines=True)
            )
        except RuntimeError:
            print(f"[WARN] No samples found. Re-loading data.")
            self._load_data_from_db()
            self._current_get_sample_retries += 1
            self.get_sample()  # CAUTION: recursion

        return json.loads(json_string)  # keys: _id, content, label

    def update_one(self, id_: str, label: str) -> None:
        """
        Update label of one single document with id.

        Args:
            id: MongoDB ID of the document as str
            label: Label to set
        """

        # # TODO
        # print(id_, label)
        # # update in memory
        # print(self.data)
        # print(self.data[self.data["_id"] == id_])

        self.data[self.data["_id"] == id_, "label"] = label

        # update DB
        self.db_connector.update_one(id=id_, label=label)

    def get_status(self):
        self._load_data_from_db()
        return self.data.groupby("label").count()

    def status_as_string(self):
        n_unlabeled = len(self.data.filter(pl.col("label").is_null()))
        n_labeled = len(self.data.filter(pl.col("label").is_not_null()))
        return f"Labeled {n_labeled:,} documents. There are {n_unlabeled:,} unlabeled documents."
