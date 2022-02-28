from src.database.mongo_connector import MongoConnector
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
