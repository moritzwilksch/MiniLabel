import json
from dataclasses import dataclass

import polars as pl
from bson.objectid import ObjectId
from pymongo import UpdateOne

from src.database.connectors import MongoConnector
from src.ml_models.base_model import MLModel


@dataclass
class Status:
    n_labeled: int
    n_total: int


class LabelingManager:
    def __init__(self, db_connector: MongoConnector, model: MLModel) -> None:
        self.db_connector = db_connector
        self.model = model

    def _add_sampling_weight(self):
        """
        Adds sampling weight "entropy" to all documents in DB.
        """
        dataf = self.db_connector.load_all_data()
        preds = self.model.predict_uncertainty(dataf)
        dataf.insert_at_idx(0, pl.Series(values=preds, name="entropy"))

        db_requests = []
        for row in dataf.to_dicts():
            db_requests.append(
                UpdateOne(
                    {"_id": ObjectId(row["_id"])}, {"$set": {"entropy": row["entropy"]}}
                )
            )

        self.db_connector.collection.bulk_write(db_requests)

    def retrain_model(self):
        """
        Loads data from DB and trains model on it
        """
        dataf = self.db_connector.load_all_data()
        self.model.fit(data=dataf)

    def get_sample(self) -> dict:
        """
        Gets the sample with highest entropy from DB.
        """
        res = list(
            self.db_connector.collection.find(
                {"label": None}, sort=[("entropy", -1)], limit=1
            )
        )[0]

        return res  # keys: _id, content, label, entropy

    def update_one(self, id_: str, label: str) -> None:
        """
        Update label of one single document with id.

        Args:
            id: MongoDB ID of the document as str
            label: Label to set
        """

        self.db_connector.update_one_label(id=id_, label=label)

    def get_status(self) -> Status:
        """
        Return job status.
        """
        return Status(
            n_labeled=self.db_connector.collection.count_documents(
                {"label": {"$ne": None}}
            ),
            n_total=self.db_connector.collection.count_documents({}),
        )

    def status_as_string(self) -> str:
        """
        Return job status as str for display in CLI
        """
        status = self.get_status()
        return f"Labeled {status.n_labeled:,} documents. There are {status.n_total - status.n_labeled:,} unlabeled documents. (Total: {status.n_total:,})"
