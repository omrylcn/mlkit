from typing import Dict
from pathlib import Path
from pydantic import BaseModel, Field


class LoggerConfig(BaseModel):
    """Configuration for logger"""

    name: str = Field(..., description="Logger name")
    level: str = Field(..., description="Logging level")
    path: Path = Field(..., description="Path to log file")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "LoggerConfig":
        return cls(**config_dict)
