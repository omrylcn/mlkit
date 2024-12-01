from typing import Dict, List,Any

from mlkit.config.data_process import FeatureCard
from mlkit.config import Config
from mlkit.log import logger


def initialize_processors(config:Config, processor_lib:Dict, pipeline_type="feature")->Dict[str, Dict[str, Any]]:
    """Initialize data processing pipeline processors based on configuration.

    Parameters
    ----------
    config : object
        Configuration object containing pipeline and processor settings.
    processor_lib : dict
        Dictionary of processor classes.
    pipeline_type : str, optional
        Type of pipeline to initialize, by default "feature".

    Returns
    -------
    dict
        Dictionary containing initialized processors for each pipeline type.
    """    
    try:
        processor_dict = {
            "online": {},
            "offline": {},
            "realtime": {},
        }

        if not hasattr(config.pipeline, pipeline_type):
            raise ValueError(f"Invalid pipeline_type: {pipeline_type}")

        pipeline_config = getattr(config.pipeline, pipeline_type).model_dump()
        processor_cards = config.data_processing.processors

        for pipeline_type in processor_dict.keys():
            if pipeline_type not in pipeline_config["steps"]:
                logger.warning(f"No steps configured for {pipeline_type} pipeline")
                continue
                
            for step in pipeline_config["steps"][pipeline_type]:
                try:
                    if step not in processor_cards:
                        raise KeyError(f"Processor card not found for step: {step}")
                    if step not in processor_lib:
                        raise KeyError(f"Processor not found in library for step: {step}")
                        
                    processor_card = processor_cards[step]
                    processor = processor_lib[step]
                    processor_object = processor(processor_card)
                    processor_dict[pipeline_type][step] = processor_object
                    logger.info(f"Initialized {pipeline_type}-{step} processor")
                except Exception as e:
                    logger.error(f"Failed to initialize {pipeline_type}-{step} processor: {str(e)}")
                    raise

        return processor_dict
    except Exception as e:
        logger.error(f"Failed to initialize processors: {str(e)}")
        raise


def check_output_columns(feature_card, data):
    """Check if output columns are in data"""
    data_columns = data.columns

    prefix = feature_card.parameters.get("prefix", "")
    prefix = prefix + "_" if len(prefix) > 0 else ""

    for o_column in feature_card.output_columns:
        if prefix + o_column not in data_columns:
            raise ValueError(f"Output column {o_column} not found in data")


def get_model_features_dict(feature_lib: Dict[str, FeatureCard], feature_col: List[str], target_col: str):
    """
    Get the feature parameters for each column in the feature library.
    """
    feature_params = {}
    target_params = {}
    for col in feature_col:
        try:
            feature_params[col] = feature_lib[col].model_dump()
        except:
            logger.error(f"Feature {col} not found in feature_lib", exc_info=True)

    try:
        target_params[target_col] = feature_lib[target_col].model_dump()
    except Exception as e:
        logger.error(f"Target {target_col} not found in feature_lib", exc_info=True)

    return feature_params, target_params
