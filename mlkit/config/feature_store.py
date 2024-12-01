from enum import Enum
from typing import Dict,List
from pydantic import BaseModel, Field

class FeatureStoreType(str,Enum):
    FEAST = "feast"

class FeatureStoreConfig(BaseModel):
    """Configuration for feature store."""

    type:FeatureStoreType = Field(..., description="Type of feature store")
    repo_path:str = Field(..., description="Path to the feature store repository")
    file_source_path:str = Field(..., description="Path to the file source")
    stored_columns: List[str] = Field(..., description="Columns to store in feature store")
    timestamp_column: str = Field(..., description="Timestamp column for feature store")
    entity_column: str = Field(..., description="Entity column for feature store")
    save_path: str = Field(..., description="Path to save features")
    ttl_days: int = Field(..., description="Time to live for features")
    



    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FeatureStoreConfig":
        return cls(**config_dict)

