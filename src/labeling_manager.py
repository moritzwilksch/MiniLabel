import json

import polars as pl

from src.database.connectors import MongoConnector
from src.ml_models.base_model import MLModel


class LabelingManager:
    def __init__(self, db_connector: MongoConnector, model: MLModel) -> None:
        self.db_connector = db_connector
        self.model = MLModel
        self.data = self.db_connector.load_all_data()

    def _add_sampling_weight(self):
        ...

    def _retrain_model(self):
        ...

    def get_sample(self):
        json_string = (
            self.data.filter(pl.col("label").is_null())
            .sample(n=1)
            .to_json(to_string=True, json_lines=True)
        )
        return json.loads(json_string)
