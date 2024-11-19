from typing import Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum


class TrainerType(str, Enum):
    SKLEARN = "sklearn"
    TIME_SERIES = "time_series"


class TrainerConfig(BaseModel):
    """Configuration for model training."""

    type: TrainerType
    validation_strategy: Dict[str, Any]
    random_state: int = Field(default=42)
    metrics: List[str]
    params: Dict[str, Any] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        return cls(**data)

    def validate(self) -> None:
        """Validate trainer configuration."""
        if not isinstance(self.validation_strategy, dict):
            raise ValueError("Validation strategy must be a dictionary")
        if not self.metrics:
            raise ValueError("At least one metric must be specified")
