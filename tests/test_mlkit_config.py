from pathlib import Path
from mlkit.config import Config
from mlkit.config.main import (
    DataConfig,
    TrackingConfig,
    LoggerConfig,
    DataProcessConfig,
    PipelineConfig,
    ModelConfig,
    TrainerConfig,
    DeployConfig,
)


def test_from_dict(sample_config_dict):
    """Test creating a Config instance from a dictionary."""
    config = Config.from_dict(sample_config_dict)
    assert isinstance(config, Config)


def test_load(config_yaml_file):
    """Test loading a Config instance from a YAML file."""
    config = Config.load(config_yaml_file)
    assert isinstance(config, Config)


def test_save(config_yaml_file,tmp_path):
    """Test saving a Config instance to a YAML file."""
    config_path = tmp_path / "config.yaml"
    config = Config.load(config_yaml_file)
    config.save(config_path)
    assert Path(config_path).exists()


def test_config_components(sample_config_dict):
    """Test if Config components are correctly instantiated."""
    config = Config.from_dict(sample_config_dict)

    assert isinstance(config.data, DataConfig)
    assert isinstance(config.tracking, TrackingConfig)
    assert isinstance(config.logger, LoggerConfig)
    assert isinstance(config.data_processing, DataProcessConfig)
    assert isinstance(config.pipeline, PipelineConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.trainer, TrainerConfig)
    assert isinstance(config.deploy, DeployConfig)


def test_save_and_load(sample_config_dict, tmp_path):
    """Test saving and loading a Config instance."""
    config_path = tmp_path / "config.yaml"
    config = Config.from_dict(sample_config_dict)
    config.save(config_path)
    loaded_config = Config.load(config_path)
    assert isinstance(loaded_config, Config)
