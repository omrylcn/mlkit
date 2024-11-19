from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class ModelRegister(str, Enum):
    MLFLOW = "mlflow"
    CUSTOM = "custom"


class DeploymentType(str, Enum):
    LOCAL = "local"
    REGISTRY = "registry"


class ModelFormat(str, Enum):
    ONNX = "onnx"
    PICKLE = "pkl"
    JOBLIB = "joblib"
    SAVEDMODEL = "savedmodel"
    TORCHSCRIPT = "pt"


class ModelConverter(str, Enum):
    ONNX = "onnx"
    JOBLIB = "joblib"


class ONNXConfig(BaseModel):
    """ONNX-specific configuration"""

    opset_version: int = Field(default=14)
    target_opset: Optional[Dict[str, int]] = Field(default=None)
    optimize: bool = Field(default=True)
    quantize: bool = Field(default=False)
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = Field(default=None)
    inference_mode: bool = Field(default=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ONNXConfig":
        return cls(**data)


class PickleConfig(BaseModel):
    """Pickle/Joblib specific configuration"""

    protocol: int = Field(default=4)
    compress: bool = Field(default=True)
    compression_level: int = Field(default=3)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PickleConfig":
        return cls(**data)


class FeatureConfig(BaseModel):
    """Feature configuration for model conversion"""

    n_features: int
    feature_names: Optional[List[str]] = Field(default=None)
    categorical_features: Optional[List[str]] = Field(default=None)
    numeric_features: Optional[List[str]] = Field(default=None)
    input_shape: Optional[tuple] = Field(default=None)
    output_shape: Optional[tuple] = Field(default=None)
    dtype: str = Field(default="float32")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureConfig":
        return cls(**data)


class MLflowRegisterConfig(BaseModel):
    """MLflow specific registration configuration"""

    tracking_uri: str
    experiment_name: str
    run_name: Optional[str] = Field(default=None)
    tags: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLflowRegisterConfig":
        return cls(**data)


class CustomRegisterConfig(BaseModel):
    """Custom registry configuration"""

    registry_uri: str
    timeout: int = Field(default=30)
    storage_path: Optional[str] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomRegisterConfig":
        return cls(**data)


class RegisterConfig(BaseModel):
    """Configuration for model registration"""

    register_type: ModelRegister
    model_name: str
    version: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    mlflow: Optional[MLflowRegisterConfig] = Field(default=None)
    custom: Optional[CustomRegisterConfig] = Field(default=None)

    model_config = {"protected_namespaces": ()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisterConfig":
        return cls(**data)


class SelectConfig(BaseModel):
    """Configuration for model selection"""

    metric_name: str
    threshold: float
    maximize: bool = Field(default=True)
    top_n: int = Field(default=5)
    model_type: Optional[str] = Field(default=None)
    task_type: Optional[str] = Field(default=None)
    additional_filters: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = {"protected_namespaces": ()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectConfig":
        return cls(**data)


class ConverterConfig(BaseModel):
    """Configuration for model conversion"""

    converter_type: ModelConverter
    target_format: ModelFormat
    features: FeatureConfig
    onnx: Optional[ONNXConfig] = Field(default=None)
    pickle: Optional[PickleConfig] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConverterConfig":
        return cls(**data)


class DeployConfig(BaseModel):
    """Main deployment configuration"""

    type: DeploymentType
    converter: ConverterConfig
    register: RegisterConfig
    select: SelectConfig

    model_config = {"protected_namespaces": ()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployConfig":
        return cls(**data)
