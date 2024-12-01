from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from feast import Entity, FeatureView, Field, FileSource, FeatureService, ValueType, FeatureStore as FeastStore
from feast.types import Float32, Int64, String, Float64, Bool
from datetime import timedelta, datetime

from mlkit.log import logger
from mlkit.data.data_manager import DF, DataManager
from mlkit.config.data_process import FeatureCard


class FeatureStore:
    def __init__(
        self,
        entity_key: str,
        timestamp_field: str,
        save_path: str,
        **kwargs,
    ):
        self.entity_key = entity_key
        self.timestamp_field = timestamp_field
        self.save_path = save_path
        self.entity = None
        self.feature_view = None
        self.feature_service = None
        self.store = None
        self.file_source_path = None
        self.file_path = None

        self.ttl_days = kwargs["ttl_days"]
        self.repo_path = kwargs["repo_path"]
        self.file_source_path = kwargs["file_source_path"]
        self.store = FeastStore(repo_path=self.repo_path)

    def validate_inputs(self, df: pd.DataFrame) -> None:
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
            "bool": Bool,
            "int64": Int64,
            "int32": Int64,
            "float64": Float64,
            "float32": Float32,
            "object": String,
            "category": String,
            "datetime64[ns]": String,
            "bool": Int64,
        }
        feast_type = type_mapping.get(str(dtype), String)
        logger.info(f"Mapping dtype {dtype} to Feast type {feast_type}")
        return feast_type

    def create_feature_store_objects(
        self, stored_features: Dict[str, FeatureCard]
    ):  # -> Tuple[Entity, FeatureView, FeatureService]:
        """Create Feast entity, feature view and feature service"""

        self.stored_columns = list(stored_features.keys())

        self.entity = Entity(
            name=self.entity_key, join_keys=[self.entity_key], description=f"Entity based on {self.entity_key}"
        )

        source = FileSource(
            path=self.file_source_path,
            timestamp_field=self.timestamp_field,
        )

        schema = []
        for column in self.stored_columns:
            feature_card = stored_features[column]
            feast_type = self._get_feast_type(feature_card.feature_type.value)
            schema.append(Field(name=column, dtype=feast_type))
            logger.debug(f"Added field {column} with type {feast_type}")

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

    def save_features(self, df: DF, stored_columns: List[str], data_manager: DataManager) -> None:
        """Save features to parquet file"""

        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        save_columns = [self.entity_key] + stored_columns

        data_manager.save(df, self.save_path, columns=save_columns)
        shape = data_manager.get_shape(df)
        logger.info(f"Saved features to {self.save_path}")
        logger.info(f"Saved {shape[0]} rows and {len(stored_columns)} features")

    def register_features(self) -> None:
        """Register feature in Feast using Python API"""
        if self.feature_service is None:
            raise ValueError("Feature service not created")

        try:
            self.store.apply([self.entity, self.feature_view, self.feature_service])
            logger.info(f"Successfully applied {self.feature_view.name} to feature store")

        except Exception as e:
            logger.error(f"Failed to apply features: {str(e)}")
            raise

    def materialize_features(self) -> None:
        """Materialize features"""
        self.store.materialize(
            start_date=datetime.now() - timedelta(days=self.ttl_days),
            end_date=datetime.now(),  # Adjust time range as needed
        )

    def get_online_features(self, entity_key: str, timestamp: str) -> pd.DataFrame:
        """Get online features"""
        raise NotImplementedError("get_online_features not implemented")
