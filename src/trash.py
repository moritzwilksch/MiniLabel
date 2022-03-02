import os
from http.client import ImproperConnectionState

from src.database.connectors import MongoConnector
from src.labeling_manager import LabelingManager
from src.ml_models.naive_bayes import NaiveBayesMLModel

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

manager = LabelingManager(db_connector=conn, model=NaiveBayesMLModel())
print(manager.get_sample())
print(manager.get_status())

manager.retrain_model()
manager._add_sampling_weight()
