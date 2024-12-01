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
from mlkit.config.feature_store import FeatureStoreConfig


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
    feature_store: FeatureStoreConfig

    model_config = {"protected_namespaces": ()}

    @classmethod
    def _convert_config_components(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert config components to their respective classes."""
        config_dict["data"] = DataConfig.from_dict(config_dict["data"])
        config_dict["tracking"] = TrackingConfig.from_dict(config_dict["tracking"])
        config_dict["logger"] = LoggerConfig.from_dict(config_dict["logger"])
        config_dict["data_processing"] = DataProcessConfig.from_dict(config_dict["data_processing"])
        config_dict["pipeline"] = PipelineConfig.from_dict(config_dict["pipeline"])
        config_dict["model"] = ModelConfig.from_dict(config_dict["model"])
        config_dict["trainer"] = TrainerConfig.from_dict(config_dict["trainer"])
        config_dict["deploy"] = DeployConfig.from_dict(config_dict["deploy"])
        config_dict["feature_store"] = FeatureStoreConfig.from_dict(config_dict["feature_store"])
        return config_dict

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file using OmegaConf."""
        conf = OmegaConf.load(path)
        config_dict = OmegaConf.to_container(conf, resolve=True)
        config_dict = cls._convert_config_components(config_dict)


        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        config_dict = cls._convert_config_components(config_dict)
        return cls(**config_dict)

    def save(self, path: Union[str, Path]) -> None:
        """Save should be instance method"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(self.model_dump_json()), path,resolve=True)
    