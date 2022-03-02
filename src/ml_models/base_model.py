from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class MLModel(ABC):
    @abstractmethod
    def preprocess(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocess data.
        """
        pass

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        """
        Fits model on data.

        Args:
            data: Polars data frame with columns "content" and "label".
        """

        ...

    @abstractmethod
    def predict_uncertainty(self, data: pl.DataFrame) -> np.ndarray:
        """
        Predict uncertainty for each data point.

        Args:
            data: Polars data frame with columns "content" and "label".

        Returns:
            Array of uncertainties, higher will be sampled first.
        """

        ...
