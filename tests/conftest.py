import sys
from pathlib import Path
import pytest
from omegaconf import OmegaConf

import yaml

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add the project root and mlkit directory to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "mlkit"))


@pytest.fixture
def sample_config_dict():
    
    return {
        "tracking": {
            "experiment_name": "test_experiment",
            "artifact_path": "",
            "tracking_uri": "sqlite:///test.db",  # Local SQLite for testing
            "log_model_path": "test_models"
        },
        "data": {
            "path": "test_data.csv",
            "file_format": "csv",
            "use_col": None,
            "options": None
        },
        "data_engine": "pandas",
        "logger": {
            "level": "INFO",
            "name": "test_logger",
            "path": "test.log"
        },
        "data_processing": {
            "steps": [
                {
                    "name": "test_datetime",
                    "description": "Test datetime processing",
                    "feature_type": "datetime",
                    "feature_class": "temporal",
                    "scope": "realtime",
                    "feature_store": True,
                    "enabled": True,
                    "input_columns": ["date_col"],
                    "output_columns": ["processed_date"]
                },
                {
                    "name": "test_categorical",
                    "description": "Test categorical processing",
                    "feature_type": "categorical",
                    "feature_class": "encoding",
                    "scope": "offline",
                    "feature_store": False,
                    "enabled": True,
                    "input_columns": ["category_col"],
                    "output_columns": ["encoded_category"],
                    "parameters": {
                        "prefix": "test",
                        "method": ["label"]
                    }
                }
            ]
        },
        "pipeline": {
            "feature": {
                "type": "feature",
                "save_type": "file",
                "feature_store": False,
                "feature_store_params": None,
                "steps": [
                    {"name": "test_datetime"},
                    {"name": "test_categorical"}
                ],
                "order": ["test_datetime", "test_categorical"],
                "stored_columns": ["processed_date", "encoded_category"],
                "timestamp_column": "date_col",
                "entity_columns": "id",
                "save_path": "test_features.parquet"
            },
            "train": {
                "type": "train",
                "load_type": "file",
                "feature_store": False,
                "load_path": "test_features.parquet",
                "data_processing": {
                    "steps": [
                        {"name": "test_categorical"}
                    ],
                    "order": ["test_categorical"]
                },
                "feature_col": [
                    "processed_date",
                    "encoded_category"
                ],
                "target": "target"
            },
            "deploy": {
                "type": "deploy",
                "metadata_schema": {
                    "config": True,
                    "data_processing_params": True,
                    "feature_params": True,
                    "target_params": True,
                    "metrics": True
                },
                "tags": None
            },
            "inference": None
        },
        "model": {
            "model_type": "lightgbm",
            "task_type": "regression",
            "model_params": {
                "n_estimators": 10,  # Reduced for testing
                "objective": "regression",
                "max_depth": 3,      # Reduced for testing
                "learning_rate": 0.1
            },
            "train_params": {
                "eval_metric": "rmse"
            },
            "random_state": 42,
            "n_splits": 2,          # Reduced for testing
            "save_path": "test_models/"
        },
        "trainer": {
            "type": "sklearn",
            "validation_strategy": {
                "validation_size": 0.2,
                "shuffle": True
            },
            "metrics": ["r2"],      # Simplified metrics
            "params": {}
        },
        "deploy": {
            "type": "local",
            "converter": {
                "converter_type": "onnx",
                "target_format": "onnx",
                "features": {
                    "n_features": 2  # Reduced for testing
                },
                "onnx": {
                    "opset_version": 13
                }
            },
            "register": {
                "register_type": "custom",  # Changed to local for testing
                "model_name": "test_model",
                "version": "0.1",
                "description": "Test model",
                "custom": {
                    "registry_uri": "sqlite:///test_registry.db",  # Local for testing
                    "timeout": 5,
                    "storage_path": "test_models/"
                }
            },
            "select": {
                "metric_name": "r2",
                "threshold": -0.2,
                "maximize": True,
                "top_n": 1
            }
        }
    }



@pytest.fixture
def config_yaml_file(sample_config_dict, tmp_path):
    """Fixture creating a temporary YAML file with the sample configuration."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path