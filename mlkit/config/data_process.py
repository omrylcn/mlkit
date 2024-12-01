from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class FeatureState(str, Enum):
    """Control states of feature transformation"""

    TRAINING = "training"  # fit_transform phase
    PREDICTION = "prediction"  # transform only phase


class FeatureType(str, Enum):
    BOOL = "bool"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    


class FeatureClass(str, Enum):
    BASIC = "basic"  # Simple transformations
    TEMPORAL = "temporal"  # Time-based features
    AGGREGATE = "aggregate"  # Aggregated features
    ENCODING = "encoding"  # Label, OneHot encoding etc
    SCALING = "scaling"  # StandardScaler, MinMaxScaler etc
    IMPUTATION = "imputation"  # Missing value handling
    TEXT = "text"  # Text processing features
    CUSTOM = "custom"  # Custom transformations


class Scope(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    REALTIME = "realtime"  # nearly same as online


class FeatureCard(BaseModel):
    """Feature ID card."""

    name: str = Field(..., description="Name of the feature")
    description: str = Field(..., description="Description of the feature")
    feature_type: FeatureType = Field(..., description="Type of the feature")
    feature_class: FeatureClass = Field(..., description="Class/category of the feature")
    scope: Scope = Field(..., description="Processing scope")
    dependencies: Optional[List[str]] = Field([], description="List of dependencies")
    processor :Optional[str] = Field(None, description="Processor name")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureCard":
        """Create FeatureCard from dictionary."""
        return cls(**data)


class ProcessorCard(BaseModel):
    """Processor configuration card."""

    name: str = Field(..., description="Name of the feature")
    input_columns: List[str] = Field(..., description="Input column names")
    output_columns: List[str] = Field(..., description="Output column names")
    scope: Scope = Field(..., description="Processing scope")
    feature_store: bool = Field(default=False, description="Whether to store in feature store")
    description: Optional[str] = Field(None, description="Feature description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessorCard":
        """Create process step from dictionary."""
        return cls(**data)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary with enum values."""
        data = super().model_dump(**kwargs)
        data["scope"] = self.scope.value
        return data


class DataProcessConfig(BaseModel):
    """Data processing configuration."""

    processors : Dict[str,ProcessorCard] = Field(..., description="Dict of process")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataProcessConfig":
        """Create DataProcessConfig from dictionary."""
        prcs = {key : ProcessorCard.from_dict(prc) for key,prc in data["processors"].items()}
        return cls(processors=prcs)
