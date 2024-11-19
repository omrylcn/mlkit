from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PipelineType(str, Enum):
    FEATURE = "feature"
    TRAIN = "train"
    INFERENCE = "inference"
    DEPLOY = "deploy"


class SaveType(str, Enum):
    FILE: str = "file"
    DATABASE: str = "database"
    S3: str = "s3"
    GCS: str = "gcs"
    AZURE: str = "azure"


class PipelineStep(BaseModel):
    name: str = Field(..., description="Name of the step")


class Pipeline(BaseModel):
    steps: List[PipelineStep] = Field(..., description="List of pipeline steps")
    order: List[str] = Field(..., description="Order of steps to execute")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """Create Pipeline from dictionary."""
        steps = [PipelineStep(name=step["name"]) for step in data["steps"]]
        return cls(steps=steps, order=data["order"])


class DataProcess(BaseModel):
    steps: List[PipelineStep] = Field(..., description="List of pipeline steps")
    order: List[str] = Field(..., description="Order of steps to execute")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataProcess":
        """Create DataProcess from dictionary."""
        steps = [PipelineStep(name=step["name"]) for step in data["steps"]]
        return cls(steps=steps, order=data["order"])


class FeaturePipeline(Pipeline):
    type: PipelineType = PipelineType.FEATURE
    save_type: SaveType
    feature_store: bool
    stored_columns: List[str] = Field(..., description="Columns to store in feature store")
    timestamp_column: str = Field(..., description="Timestamp column for feature store")
    entity_columns: str = Field(..., description="Entity column for feature store")
    save_path: str = Field(..., description="Path to save features")
    feature_store_params: Optional[Dict[str, Any]] = Field(None, description="Parameters for feature store")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeaturePipeline":
        """Create FeaturePipelineStep from dictionary."""
        steps = [PipelineStep(name=step["name"]) for step in data["steps"]]
        return cls(
            steps=steps,
            order=data["order"],
            type=data["type"],
            save_type=data["save_type"],
            feature_store=data["feature_store"],
            stored_columns=data["stored_columns"],
            timestamp_column=data["timestamp_column"],
            entity_columns=data["entity_columns"],
            save_path=data["save_path"],
            feature_store_params=data["feature_store_params"],
        )


class TrainPipeline(BaseModel):
    type: PipelineType = PipelineType.TRAIN
    load_type: SaveType
    feature_store: bool
    load_path: str = Field(..., description="Path to load features")
    data_processing: DataProcess = Field(..., description="Data processing steps")
    feature_col: List[str] = Field(..., description="Columns to use")
    target: str = Field(..., description="Target column")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainPipeline":
        """Create TrainPipeline from dictionary."""

        data_process = DataProcess.from_dict(data["data_processing"])
        return cls(
            type=data["type"],
            load_type=data["load_type"],
            feature_store=data["feature_store"],
            load_path=data["load_path"],
            data_processing=data_process,
            feature_col=data["feature_col"],
            target=data["target"],
        )


class DeployPipeline(BaseModel):
    type: PipelineType = PipelineType.DEPLOY
    metadata_schema: Dict[str, bool] = Field(..., description="Metadata schema")
    tags: Optional[Dict[str, Any]] = Field(None, description="Tags for the model")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployPipeline":
        """Create DeployPipeline from dictionary."""
        return cls(
            type=data["type"],
            metadata_schema=data["metadata_schema"],
        )


class PipelineConfig(BaseModel):
    feature: Optional[FeaturePipeline] = Field(None, description="Feature pipeline configuration")
    train: Optional[TrainPipeline] = Field(None, description="Training pipeline configuration")
    inference: Optional[Pipeline] = Field(None, description="Inference pipeline configuration")
    deploy: Optional[DeployPipeline] = Field(None, description="Deployment pipeline configuration")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from dictionary."""
        feature_pipeline = FeaturePipeline.from_dict(data["feature"]) if data["feature"] else None
        train_pipeline = TrainPipeline.from_dict(data["train"]) if data["train"] else None

        inference_pipeline = Pipeline.from_dict(data["inference"]) if data["inference"] else None
        deploy_pipeline = DeployPipeline.from_dict(data["deploy"]) if data["deploy"] else None
        return cls(feature=feature_pipeline, train=train_pipeline, inference=inference_pipeline, deploy=deploy_pipeline)
