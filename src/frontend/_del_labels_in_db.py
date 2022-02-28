import os
from pymongo import MongoClient

user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")

client = MongoClient(
    f"mongodb://{user}:{password}@157.90.167.200:27017", authSource="admin"
)
db = client["data_labeling"]
collection = db["dev_coll"]

collection.update_many({}, {"$set": {"label": None}})
print("Resetted labels.")
