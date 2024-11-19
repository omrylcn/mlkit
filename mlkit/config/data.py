from typing import Dict, Optional, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data loading."""

    path: Path = Field(..., description="Path to the data file")
    file_format: str = Field(..., description="Format of the data file (e.g., csv, parquet)")
    use_col: Optional[List[str]] = Field(None, description="List of columns to use")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for data loading")

    @field_validator("path")
    def validate_path(cls, v):
        if not isinstance(v, Path):
            v = Path(v)
        return v

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "DataConfig":
        return cls(**config_dict)
