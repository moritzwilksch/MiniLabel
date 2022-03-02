from src.ml_models.base_model import MLModel
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np


class NaiveBayesMLModel(MLModel):
    def preprocess(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(pl.col("label").is_not_null())

    def fit(self, data: pl.DataFrame) -> None:
        data = self.preprocess(data).to_pandas()

        pipeline = Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())])

        pipeline.fit(data["content"], data["label"])

        with open("models/nb_model.joblib", "wb") as f:
            joblib.dump(pipeline, f)

    def predict_uncertainty(self, data: pl.DataFrame) -> np.ndarray:
        pipeline = joblib.load("models/nb_model.joblib")

        data = data.to_pandas()

        return self.entropy(pipeline.predict_proba(data["content"]))
