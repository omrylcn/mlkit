from typing import Dict, Any
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class ProcessorType(str, Enum):
    MISSING = "missing"
    CATEGORICAL = "categorical"
    SCALE = "scale"
    SELECT = "select"


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class TrainerType(str, Enum):
    SKLEARN = "sklearn"
    TIME_SERIES = "time_series"


class DataEngine(str, Enum):
    POLARS = "polars"
    PANDAS = "pandas"


class ModelConfig(BaseModel):
    """Configuration for model setup and training."""

    model_type: ModelType = Field(..., description="Type of model to use")
    task_type: TaskType = Field(..., description="Type of ML task")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    train_params: Dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(default=0.2, description="Proportion of data for testing")
    n_splits: int = Field(default=5, description="Number of cross-validation splits")
    save_path: Path = Field(default=Path("models/"), description="Path to save model artifacts")

    model_config = {"protected_namespaces": ()}

    @field_validator("test_size")
    def validate_test_size(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"Test size must be between 0 and 1, got {v}")
        return v

    @field_validator("n_splits")
    def validate_n_splits(cls, v):
        if v < 2:
            raise ValueError(f"Number of splits must be at least 2, got {v}")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(**data)
