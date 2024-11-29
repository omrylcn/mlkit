import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from feast import Entity, FeatureView, Field, FileSource, FeatureService, ValueType, FeatureStore as FeastStore
from feast.types import Float32, Int64, String
from datetime import timedelta, datetime

from pathlib import Path

from mlkit.log import logger


class FeatureStore:
    def __init__(
        self,
        entity_key: str,
        timestamp_field: str,
        save_path: str,
        feature_store_params: Optional[Dict[str, Any]] = None,
    ):
        self.entity_key = entity_key
        self.timestamp_field = timestamp_field
        self.save_path = save_path
        self.entity = None
        self.feature_view = None
        self.feature_service = None
        self.store = None
        self.file_source_path = None
        self.ttl_days = None
        self.file_path = None

        if feature_store_params:
            self.store = FeastStore(repo_path=feature_store_params["repo_path"])
            self.file_source_path = feature_store_params["file_source_path"]

    def _validate_inputs(self, df) -> None:
        """Validate input data and parameters"""

        # Check timestamp field type
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_field]):
            logger.warning(f"Converting {self.timestamp_field} to datetime")
            try:
                df[self.timestamp_field] = pd.to_datetime(df[self.timestamp_field])
            except Exception as e:
                raise ValueError(f"Could not convert {self.timestamp_field} to datetime: {e}")

        # Validate stored columns
        valid_columns = []
        for col in self.stored_columns:
            if col in df.columns:
                valid_columns.append(col)
            else:
                logger.warning(f"Column {col} not found in DataFrame")

        if not valid_columns:
            raise ValueError("No valid columns to store")

        self.stored_columns = valid_columns

        missing_stats = df[self.stored_columns].isnull().sum()
        if missing_stats.any():
            logger.warning(f"Missing values detected:\n{missing_stats[missing_stats > 0]}")

    def _get_feast_type(self, dtype: np.dtype) -> ValueType:
        """Map pandas dtypes to Feast types"""
        type_mapping = {
            "int64": Int64,
            "int32": Int64,
            "float64": Float32,
            "float32": Float32,
            "object": String,
            "category": String,
            "datetime64[ns]": String,
            "bool": Int64,
        }
        feast_type = type_mapping.get(str(dtype), String)
        logger.debug(f"Mapping dtype {dtype} to Feast type {feast_type}")
        return feast_type

    def create_feature_store_objects(self, df: pd.DataFrame, stored_columns: str, validate_data: bool = True, ttl_days: int = 365):  # -> Tuple[Entity, FeatureView, FeatureService]:
        """Create Feast entity, feature view and feature service"""

        self.stored_columns = stored_columns
        self.ttl_days = ttl_days

        if validate_data:
            self._validate_inputs(df)

        self.entity = Entity(name=self.entity_key, join_keys=[self.entity_key], description=f"Entity based on {self.entity_key}")

        source = FileSource(
            path=self.file_source_path,
            timestamp_field=self.timestamp_field,
        )

        schema = []
        for column in self.stored_columns + [self.timestamp_field]:
            if column in df.columns:
                feast_type = self._get_feast_type(df[column].dtype)
                schema.append(Field(name=column, dtype=feast_type))
                logger.debug(f"Added field {column} with type {feast_type}")
            else:
                logger.warning(f"Column {column} not found in DataFrame")

        self.feature_view = FeatureView(
            name=f"{self.entity_key}_feature_view",
            entities=[self.entity],
            schema=schema,
            source=source,
            online=True,
            ttl=timedelta(days=self.ttl_days),
            description=f"Feature view for {self.entity_key} with {len(schema)} features",
        )

        self.feature_service = FeatureService(name="customer_feature_service", features=[self.feature_view])
        logger.info("Create feature store objects")

        # return self.entity, self.feature_view, self.feature_service

    def save_features(self, df: pd.DataFrame, stored_columns: List[str]) -> None:
        """Save features to parquet file"""

        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        save_columns = [self.entity_key, self.timestamp_field] + stored_columns
        df_to_save = df[save_columns].copy()

        df_to_save.to_parquet(self.save_path, index=False)
        logger.info(f"Saved features to {self.save_path}")

        logger.info(f"Saved {len(df_to_save)} rows and {len(stored_columns)} features")

    def register_features(self) -> None:
        """Register feature in Feast using Python API"""
        if self.feature_service is None:
            raise ValueError("Feature service not created")

        try:
            # Initialize feature store
            # store = FeastStore(repo_path=self.feature_store_params["repo_path"])

            # Apply feature definitions
            self.store.apply([self.entity, self.feature_view, self.feature_service])

            logger.info(f"Successfully applied {self.feature_view.name} to feature store")
        except Exception as e:
            logger.error(f"Failed to apply features: {str(e)}")
            raise

    def materialize_features(self) -> None:
        """Materialize features"""
        self.store.materialize(
            start_date=datetime.now() - timedelta(days=720),
            end_date=datetime.now(),  # Adjust time range as needed
        )

    def get_online_features(self, entity_key: str, timestamp: str) -> pd.DataFrame:
        """Get online features"""
        if self.feature_service is None:
            raise ValueError("Feature service not created")

    # def register_features(self) -> None:
    #     """Register feature in Feast"""
    #     if self.feature_service is None:
    #         raise ValueError("Feature service not created")

    #     import os
    #     import subprocess

    #     old_dir = os.getcwd()
    #     os.chdir(self.feature_store_params["repo_path"])
    #     subprocess.run(["feast", "apply"])

    #     #fs = FeastStore(self.feature_store_params["repo_path"])
    #     #fs.apply([self.entity, self.feature_view, self.feature_service],partial=False)
    #     os.chdir(old_dir)

    #     logger.info(f"Applied {self.feature_view.name} to DataFrame")
