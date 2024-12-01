from typing import Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
import pandas as pd
import polars as pl
from pathlib import Path
import pickle

from mlkit.config.data_process import ProcessorCard, FeatureState


class BaseFeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """Base class for all feature transformers with state control"""

    def __init__(self, processor_card: ProcessorCard):
        self.processor_card = processor_card
        self._fitted_state: Dict[str, Any] = {}
        self.state = FeatureState.TRAINING
        self.save_object = None

    def set_state(self, state: FeatureState):
        """Set the current state of feature transformer"""
        self.state = state

    def _check_fit_state(self):
        """Verify if fitting is allowed in current state"""
        if self.state != FeatureState.TRAINING:
            raise ValueError("Fit only allowed in TRAINING state")

    @abstractmethod
    def fit(self, X, y=None):
        """Fit transformer to data with state control"""
        self._check_fit_state()
        pass

    @abstractmethod
    def transform(self, X):
        """Transform data"""
        pass

    @abstractmethod
    def check_after_process(self, X):
        """Check check data result"""
        pass

    @abstractmethod
    def check_before_process(self, X):
        """Check check input data"""
        pass

    # get attributes
    @property
    def saving_params(self):
        """Return parameters for saving"""
        return {
            "processor_card": self.processor_card.model_dump(),
            "fitted_state": self._fitted_state,
            "state": self.state.value,  # Also save the state
        }

    def load_params(self, params: Dict[str, Any]):
        """Load parameters from a dictionary"""
        self.processor_card = ProcessorCard(**params["processor_card"])
        self._fitted_state = params["fitted_state"]
        self.state = FeatureState(params["state"])

    def save(self, path: Union[Path, str]) -> None:
        """Save transformer state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.save_object = self.saving_params

        with open(path / "save_object.pkl", "wb") as f:
            pickle.dump(self.save_object, f)

    def load(self, path: Union[Path, str]) -> None:
        """Load transformer state"""
        path = Path(path)

        with open(path / "save_object.pkl", "rb") as f:
            self.save_object = pickle.load(f)
            self.processor_card = ProcessorCard.from_dict(self.save_object["processor_card"])
            self._fitted_state = self.save_object["fitted_state"]
            self.state = FeatureState(self.save_object["state"])  # Restore the state


class PandasProcessor(BaseFeatureTransformer):
    """Base class for Pandas-based transformers"""

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PandasProcessor requires pandas DataFrame")
        self._check_fit_state()  # Add state check
        return self._fit_impl(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PandasProcessor requires pandas DataFrame")
        return self._transform_impl(X)

    @abstractmethod
    def _fit_impl(self, X: pd.DataFrame, y=None):
        """Implementation of pandas-specific fit logic"""
        pass

    @abstractmethod
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Implementation of pandas-specific transform logic"""
        pass

    def check_after_process(self, X: pd.DataFrame):
        """Check check data result"""
        
        for column in self.processor_card.output_columns:
            if column not in X.columns:
                raise ValueError(f"Column {column} not found in DataFrame")
        
    def check_before_process(self, X: pd.DataFrame):
        """Check check input data"""
        for column in self.processor_card.input_columns:
            if column not in X.columns:
                raise ValueError(f"Column {column} not found in DataFrame")


class PolarsProcessor(BaseFeatureTransformer):
    """Base class for Polars-based transformers"""

    def fit(self, X: pl.DataFrame, y=None):
        if not isinstance(X, pl.DataFrame):
            raise TypeError("PolarsProcessor requires polars DataFrame")
        self._check_fit_state()  # Add state check
        return self._fit_impl(X, y)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(X, pl.DataFrame):
            raise TypeError("PolarsProcessor requires polars DataFrame")
        return self._transform_impl(X)

    @abstractmethod
    def _fit_impl(self, X: pl.DataFrame, y=None):
        """Implementation of polars-specific fit logic"""
        pass

    @abstractmethod
    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        """Implementation of polars-specific transform logic"""
        pass


class CategoricalTransformPandas(PandasProcessor):
    def __init__(self, processor_card: ProcessorCard):
        super().__init__(processor_card)
        self.processor_card = processor_card

        if len(self.processor_card.input_columns) != len(self.processor_card.parameters["method"]):
            raise ValueError("Input columns and method length must match")

        self._category_maps = {}

    def _fit_impl(self, X: pd.DataFrame, y=None):
        missing_cols = [col for col in self.processor_card.input_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        for col, method in zip(self.processor_card.input_columns, self.processor_card.parameters["method"]):
            unique_values = X[col].unique()
            if method == "label":
                self._category_maps[col] = {cat: idx for idx, cat in enumerate(unique_values)}
            elif method == "onehot":
                self._category_maps[col] = unique_values
            else:
                raise ValueError(f"Unsupported method: {method}. Use 'label' or 'onehot'")

            self._fitted_state["_category_maps"] = self._category_maps

        return self

    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        prefix = self.processor_card.parameters["prefix"]
        self._category_maps = self._fitted_state["_category_maps"]

        for col, method in zip(self.processor_card.input_columns, self.processor_card.parameters["method"]):
            if method == "label":
                output_col = f"{prefix}_{col}"
                df[output_col] = df[col].map(self._category_maps[col])
                df[output_col] = df[output_col].fillna(-1).astype(int)

                if col not in self.processor_card.output_columns:
                    df = df.drop(columns=[col])

            elif method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=f"{prefix}_{col}")
                df = pd.concat([df, dummies], axis=1)

                if col not in self.processor_card.output_columns:
                    df = df.drop(columns=[col])
            else:
                raise ValueError(f"Unsupported method: {method}. Use 'label' or 'onehot'")

        return df


class ScalerTransformPandas(PandasProcessor):
    def __init__(self, processor_card: ProcessorCard, feature_store=None):
        super().__init__(processor_card, feature_store)
        if not all(method == "standard" for method in self.processor_card.parameters["method"]):
            raise ValueError("All methods must be 'standard' for ScalerTransform")

    def _fit_impl(self, X: pd.DataFrame, y=None):
        self._fitted_state["mean"] = X[self.processor_card.input_columns].mean().to_dict()
        self._fitted_state["std"] = X[self.processor_card.input_columns].std().to_dict()
        return self

    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        prefix = self.processor_card.parameters["prefix"]

        for col in self.processor_card.input_columns:
            output_col = f"{prefix}_{col}"
            df[output_col] = (df[col] - self._fitted_state["mean"][col]) / self._fitted_state["std"][col]

        return df


class MissingValueProcessorPandas(PandasProcessor):
    def __init__(self, processor_card: ProcessorCard, feature_store=None):
        super().__init__(processor_card, feature_store)
        valid_methods = ["mean", "median", "mode", "constant"]
        if not all(method in valid_methods for method in self.processor_card.parameters["method"]):
            raise ValueError(f"Methods must be one of {valid_methods}")

    def _fit_impl(self, X: pd.DataFrame, y=None):
        self._fitted_state["fill_values"] = {}

        for col, method in zip(self.processor_card.input_columns, self.processor_card.parameters["method"]):
            if X[col].dtype == "float64" and method in ["mean", "median"]:
                if method == "mean":
                    self._fitted_state["fill_values"][col] = float(X[col].mean())
                else:  # median
                    self._fitted_state["fill_values"][col] = float(X[col].median())
            elif method == "mode":
                self._fitted_state["fill_values"][col] = X[col].mode()[0]
            elif method == "constant":
                self._fitted_state["fill_values"][col] = self.processor_card.parameters["fill_value"]
            else:
                raise ValueError(f"Invalid method {method} for column type {X[col].dtype}")

        return self

    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in self.processor_card.input_columns:
            fill_value = self._fitted_state["fill_values"][col]
            df[col] = df[col].fillna(fill_value)

        return df
