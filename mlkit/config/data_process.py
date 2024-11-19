from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class FeatureState(str, Enum):
    """Control states of feature transformation"""

    TRAINING = "training"  # fit_transform phase
    PREDICTION = "prediction"  # transform only phase


class FeatureType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    CATEGORICAL = "categorical"
    TEMPORAL = "datetime"
    MIXED = "mixed"


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureCard":
        """Create FeatureCard from dictionary."""
        return cls(**data)


class ProcessStepCard(BaseModel):
    """Process step configuration card."""

    name: str = Field(..., description="Name of the feature")
    feature_type: FeatureType = Field(..., description="Type of the feature")
    feature_class: FeatureClass = Field(..., description="Class/category of the feature")
    input_columns: List[str] = Field(..., description="Input column names")
    output_columns: List[str] = Field(..., description="Output column names")
    scope: Scope = Field(..., description="Processing scope")
    feature_store: bool = Field(default=False, description="Whether to store in feature store")
    enabled: bool = Field(default=True, description="Whether the feature is enabled")
    description: Optional[str] = Field(None, description="Feature description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    model_config = {"frozen": True}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessStepCard":
        """Create process step from dictionary."""
        return cls(**data)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary with enum values."""
        data = super().model_dump(**kwargs)
        data["feature_type"] = self.feature_type.value
        data["feature_class"] = self.feature_class.value
        data["scope"] = self.scope.value
        return data


class DataProcessConfig(BaseModel):
    """Data processing configuration."""

    steps: List[ProcessStepCard] = Field(..., description="List of feature cards")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataProcessConfig":
        """Create DataProcessConfig from dictionary."""
        steps = [ProcessStepCard.from_dict(step) for step in data["steps"]]
        return cls(steps=steps)
