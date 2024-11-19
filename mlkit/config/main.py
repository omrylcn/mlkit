from typing import Union, Dict, Any
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from omegaconf import OmegaConf

from mlkit.config.data import DataConfig
from mlkit.config.track import TrackingConfig
from mlkit.config.log import LoggerConfig
from mlkit.config.data_process import DataProcessConfig
from mlkit.config.pipeline import PipelineConfig
from mlkit.config.model import ModelConfig
from mlkit.config.train import TrainerConfig
from mlkit.config.deploy import DeployConfig


class DataEngine(str, Enum):
    POLARS = "polars"
    PANDAS = "pandas"


class Config(BaseModel):
    """Main configuration class."""

    data: DataConfig
    data_engine: DataEngine = Field(default=DataEngine.PANDAS, description="Data processing engine")
    data_processing: DataProcessConfig
    tracking: TrackingConfig
    logger: LoggerConfig
    pipeline: PipelineConfig
    model: ModelConfig
    trainer: TrainerConfig
    deploy: DeployConfig

    model_config = {"protected_namespaces": ()}

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file using OmegaConf."""
        # Load the YAML file using OmegaConf
        conf = OmegaConf.load(path)
        config_dict = OmegaConf.to_container(conf, resolve=True)

        config_dict["data"] = DataConfig.from_dict(config_dict["data"])
        config_dict["tracking"] = TrackingConfig.from_dict(config_dict["tracking"])
        config_dict["logger"] = LoggerConfig.from_dict(config_dict["logger"])
        config_dict["data_processing"] = DataProcessConfig.from_dict(config_dict["data_processing"])
        config_dict["pipeline"] = PipelineConfig.from_dict(config_dict["pipeline"])
        config_dict["model"] = ModelConfig.from_dict(config_dict["model"])
        config_dict["trainer"] = TrainerConfig.from_dict(config_dict["trainer"])
        config_dict["deploy"] = DeployConfig.from_dict(config_dict["deploy"])

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def save(cls, path: Union[str, Path], config: "Config") -> None:
        """Save configuration to a YAML file using OmegaConf."""
        OmegaConf.save(OmegaConf.create(config.model_dump()), path)
