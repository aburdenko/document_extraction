#!/usr/bin/env python3

import os
import argparse
from google.cloud import aiplatform

# Index settings
DIMENSIONS = 768  # ASSUMPTION: Based on 'text-embedding-004' model. Change if needed.
APPROXIMATE_NEIGHBORS = 150 # ASSUMPTION: A common default for this required field.
SHARD_SIZE = "SHARD_SIZE_MEDIUM"

# Advanced Options
DISTANCE_MEASURE = "DOT_PRODUCT_DISTANCE"
FEATURE_NORM = "NONE"


def create_vector_search_index(
    project_id: str,
    region: str,
    display_name: str,
    contents_delta_uri: str,
    dimensions: int,
    approximate_neighbors_count: int,
    shard_size: str,
    distance_measure_type: str,
    feature_norm_type: str,
):
    """
    Creates a Vertex AI MatchingEngineIndex.
    """
    print(f"Initializing Vertex AI for project '{project_id}' in region '{region}'...")
    aiplatform.init(project=project_id, location=region)

    print(f"Starting creation of index '{display_name}'...")
    # UPDATED THIS LINE: Replaced .create() with .create_tree_ah_index()
    my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        contents_delta_uri=contents_delta_uri,
        dimensions=dimensions,
        approximate_neighbors_count=approximate_neighbors_count,
        distance_measure_type=distance_measure_type,
        feature_norm_type=feature_norm_type,
        shard_size=shard_size,
    )

    print("---" * 10)
    print("✅ Index creation initiated successfully!")
    print(f"   Display Name: {my_index.display_name}")
    print(f"   Resource Name: {my_index.resource_name}")
    print("You can monitor the creation status in the Google Cloud Console.")
    print("---" * 10)
    return my_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Vertex AI Vector Search Index.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Fetch defaults from environment variables set by configure.sh
    parser.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"), help="Google Cloud project ID.")
    parser.add_argument("--region", type=str, default=os.getenv("REGION", "us-central1"), help="Google Cloud region.")
    parser.add_argument("--display-name", type=str, default=os.getenv("INDEX_DISPLAY_NAME"), help="Display name for the Vector Search Index.")
    parser.add_argument("--staging-gcs-bucket", type=str, default=os.getenv("STAGING_GCS_BUCKET"), help="GCS bucket for staging index files.")

    args = parser.parse_args()

    # --- Validate Configuration ---
    if not args.project_id:
        print("ERROR: Project ID is not set. Please provide it via the --project-id flag or set the PROJECT_ID environment variable.")
        exit(1)
    if not args.display_name:
        print("ERROR: Index display name is not set. Please provide it via the --display-name flag or set the INDEX_DISPLAY_NAME environment variable.")
        exit(1)
    if not args.staging_gcs_bucket:
        print("ERROR: Staging GCS bucket is not set. Please provide it via the --staging-gcs-bucket flag or set the STAGING_GCS_BUCKET environment variable.")
        exit(1)

    # The contents_delta_uri needs to be a GCS folder.
    contents_delta_uri = f"gs://{args.staging_gcs_bucket}/vector_search_index_contents"

    create_vector_search_index(
        project_id=args.project_id,
        region=args.region,
        display_name=args.display_name,
        contents_delta_uri=contents_delta_uri,
        dimensions=DIMENSIONS,
        approximate_neighbors_count=APPROXIMATE_NEIGHBORS,
        shard_size=SHARD_SIZE,
        distance_measure_type=DISTANCE_MEASURE,
        feature_norm_type=FEATURE_NORM,
    )