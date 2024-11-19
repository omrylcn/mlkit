# run_pipeline.py
import sys
import argparse
import time
import requests
import os
import logging

from prefect import serve
from projects.purchase_prediction.pipeline import train_pipeline, feature_pipeline,deploy_pipeline,materialize_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wait_for_prefect():
    """Simple wait for Prefect server"""
    prefect_url = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
    health_url = f"{prefect_url}/health"

    for i in range(6):  # Try for 30 seconds
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                logger.info("Prefect server is ready!")
                return True
        except:
            pass

        logger.info("Waiting for Prefect server...")
        time.sleep(5)

    raise RuntimeError("Prefect server not available")


def get_pipeline(pipeline_type: str):
    """Get pipeline function"""
    pipeline_map = {
        "feature": feature_pipeline,
        "train": train_pipeline,
        "deploy": deploy_pipeline,
        "materialize": materialize_pipeline,
    }
    return pipeline_map[pipeline_type]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["feature", "train", "deploy","materialize"],
        help="Pipeline type to run",
    )
    
    parser.add_argument('--wait_prefect',
                       type=str,
                       choices=['true', 'false'],
                       default='true',
                       help='Verbose output (true/false)')
    
    parser.add_argument("--all",
                        type=str,
                        choices=['true', 'false'],
                        default='false',
                        help='Server all pipelines (true/false)')
    
    parser.add_argument('--deploy',
                        type=str,
                        choices=['true', 'false'],
                        default='true',
                        help='Deploy pipeline (true/false)')

    args = parser.parse_args()
    
    try:
        if args.all == 'true':
            deployments = [
                get_pipeline(p).to_deployment(name=f"{p}_pipeline")
                for p in ["train", "feature", "deploy", "materialize"]
            ]
            serve(*deployments)
        else:
            if args.wait_prefect=='true':
                wait_for_prefect()

            pipeline = get_pipeline(args.pipeline)
            if args.deploy == 'true':
               
                pipeline_deployment = pipeline.to_deployment(name=f"{args.pipeline}_pipeline")
                serve(pipeline_deployment)

            else:
                pipeline()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
