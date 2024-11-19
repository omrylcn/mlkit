from typing import Dict, Optional
from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    """Configuration for experiment tracking."""

    experiment_name: str = Field(..., description="Name of the experiment")
    artifact_path: str = Field(..., description="Path to store artifacts")
    tracking_uri: Optional[str] = Field(None, description="URI for tracking server")
    log_model_path: Optional[str] = Field("models", description="Path to log models")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TrackingConfig":
        return cls(**config_dict)
