#!/usr/bin/env python3
"""
This script creates a Vertex AI Vector Store (Matching Engine Index) from
unstructured text files stored in a Google Cloud Storage bucket.

It performs the following steps:
1. Reads text files from a source GCS bucket.
2. Generates embeddings for the text content using a Vertex AI Embedding Model.
3. Formats the embeddings into the required JSONL format.
4. Uploads the JSONL file to a staging GCS bucket.
5. Creates a new Matching Engine Index from the embeddings data.
6. Creates a new Index Endpoint to serve the index.
7. Deploys the new Index to the new Endpoint.
"""

import argparse
import logging
import os
import json
import time
import io
import sys
import uuid

import vertexai
from pypdf import PdfReader
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constants ---
# The text-embedding-gecko models have an output dimension of 768.
# https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embedding
EMBEDDING_DIMENSIONS = 768
# How many neighbors to find for approximate search.
APPROXIMATE_NEIGHBORS_COUNT = 15
# A unique ID for the deployed index.
DEPLOYED_INDEX_ID_PREFIX = "vs_deployed"
# --- Text Chunking Configuration ---
CHUNK_SIZE = 2000  # Characters per chunk
CHUNK_OVERLAP = 200 # Characters to overlap between chunks
# --- Embedding API Configuration ---
# The `text-embedding-004` model has a limit of 20,000 tokens per request.
# We set a smaller batch size to ensure the total tokens in a batch do not exceed this limit.
EMBEDDING_BATCH_SIZE = 20 # Number of text chunks to send in a single API call.


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits text into chunks of a specified size with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def process_gcs_bucket(
    source_gcs_bucket_name: str,
) -> list[str]:
    """
    Reads and chunks files from a GCS bucket, handling PDF, TXT, and CSV.
    Returns their content as a list of text chunks.
    """
    logger.info(f"Processing files from GCS bucket: gs://{source_gcs_bucket_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(source_gcs_bucket_name)
    blobs = bucket.list_blobs()

    texts = []
    for blob in blobs:
        content = ""
        try:
            logger.info(f"  - Reading file: {blob.name}")
            file_name_lower = blob.name.lower()

            if file_name_lower.endswith('.pdf'):
                pdf_bytes = blob.download_as_bytes()
                pdf_file = io.BytesIO(pdf_bytes)
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    content += page.extract_text() or ""  # Ensure content is not None
                logger.info(f"    Successfully extracted text from PDF: {blob.name}")
            elif file_name_lower.endswith(('.txt', '.csv', '.md', '.json', '.jsonl')):
                content = blob.download_as_text()
                logger.info(f"    Successfully read text file: {blob.name}")
            else:
                logger.warning(f"    Skipping unsupported file type: {blob.name}")
                continue

            if content:
                chunked_content = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
                texts.extend(chunked_content)
                logger.info(f"    Chunked content into {len(chunked_content)} chunks.")

        except Exception as e:
            logger.error(f"Could not process blob '{blob.name}'. Skipping. Error: {e}", exc_info=True)

    if not texts:
        raise ValueError(f"No processable text files found in gs://{source_gcs_bucket_name}")

    logger.info(f"Successfully processed and chunked all files into {len(texts)} total text chunks.")
    return texts


def generate_embeddings(
    texts: list[str],
    embedding_model_name: str,
) -> list[list[float]]:
    """Generates embeddings for a list of text strings."""
    logger.info(f"Generating embeddings using model: {embedding_model_name}")
    model = TextEmbeddingModel.from_pretrained(embedding_model_name)
    # The API supports batching, but the total token count per request is limited.
    # We use a smaller batch size to avoid exceeding the token limit.
    embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        logger.info(f"  - Embedding batch {i//EMBEDDING_BATCH_SIZE + 1} of {len(texts)//EMBEDDING_BATCH_SIZE + 1}...")
        embeddings.extend([e.values for e in model.get_embeddings(batch)])
    logger.info(f"Successfully generated {len(embeddings)} embeddings.")
    return embeddings


def format_and_upload_embeddings(
    embeddings: list[list[float]],
    staging_gcs_bucket_name: str,
) -> str:
    """
    Formats embeddings into JSONL and uploads to a GCS bucket.
    Returns the GCS URI of the directory containing the embeddings file.
    """
    # Create a unique directory for this batch of embeddings.
    batch_id = uuid.uuid4()
    embeddings_dir_uri = f"gs://{staging_gcs_bucket_name}/embeddings/{batch_id}"
    embeddings_file_uri = f"{embeddings_dir_uri}/embeddings.json"

    logger.info(f"Formatting and uploading embeddings to {embeddings_file_uri}")

    # Create the JSONL content.
    jsonl_content = "\n".join(
        [
            json.dumps({"id": str(i), "embedding": emb})
            for i, emb in enumerate(embeddings)
        ]
    )

    # Upload to GCS.
    storage_client = storage.Client()
    bucket_name, blob_name = embeddings_file_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(jsonl_content)

    logger.info("Upload complete.")
    return embeddings_dir_uri


def create_matching_engine_index(
    project_id: str,
    location: str,
    display_name: str,
    embeddings_gcs_uri: str,
) -> aiplatform.MatchingEngineIndex:
    """Creates a new Matching Engine Index."""
    logger.info(f"Creating new Matching Engine Index: '{display_name}'")
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        project=project_id,
        location=location,
        display_name=display_name,
        contents_delta_uri=embeddings_gcs_uri,
        dimensions=EMBEDDING_DIMENSIONS,
        approximate_neighbors_count=APPROXIMATE_NEIGHBORS_COUNT,
        sync=True,  # Wait for the operation to complete.
    )
    logger.info(f"Index created successfully. ID: {index.resource_name}")
    return index


def create_index_endpoint(
    project_id: str,
    location: str,
    display_name: str,
) -> aiplatform.MatchingEngineIndexEndpoint:
    """Creates a new Index Endpoint."""
    logger.info(f"Creating new Index Endpoint: '{display_name}'")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        project=project_id,
        location=location,
        display_name=display_name,
        public_endpoint_enabled=True,
    )
    logger.info(f"Endpoint created successfully. ID: {endpoint.resource_name}")
    return endpoint


def main(args):
    """Main execution function."""
    try:
        # --- Configuration Setup ---
        # Prioritize command-line arguments, then fall back to environment variables.
        project_id = args.project_id or os.getenv("PROJECT_ID")
        region = args.region or os.getenv("REGION", "us-central1")
        source_bucket = args.source_gcs_bucket or os.getenv("SOURCE_GCS_BUCKET")
        staging_bucket = args.staging_gcs_bucket or os.getenv("STAGING_GCS_BUCKET")
        index_display_name_base = args.index_display_name or os.getenv("INDEX_DISPLAY_NAME", "my-vector-store-index")
        endpoint_display_name_base = args.index_endpoint_display_name or os.getenv("INDEX_ENDPOINT_DISPLAY_NAME", "my-vector-store-endpoint")
        embedding_model_name = args.embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME", "textembedding-gecko@003")

        # --- Validate Configuration ---
        if not project_id or "your-gcp-project-id-here" in project_id:
            logger.error("Project ID is not set. Please provide it via the --project_id flag or by editing .scripts/configure.sh")
            sys.exit(1)
        
        if not source_bucket or "your-source-bucket-name-here" in source_bucket:
            logger.error("Source GCS bucket is not set. Please provide it via the --source_gcs_bucket flag or by editing .scripts/configure.sh")
            sys.exit(1)

        if not staging_bucket or "your-staging-bucket-name-here" in staging_bucket:
            logger.error("Staging GCS bucket is not set. Please provide it via the --staging_gcs_bucket flag or by editing .scripts/configure.sh")
            sys.exit(1)

        # --- Initialization ---
        logger.info(f"Initializing Vertex AI for project '{project_id}' in region '{region}'...")
        vertexai.init(project=project_id, location=region)
        aiplatform.init(project=project_id, location=region)

        # --- Step 1: Process source files ---
        texts = process_gcs_bucket(source_bucket)

        # --- Step 2: Generate Embeddings ---
        embeddings = generate_embeddings(texts, embedding_model_name)

        # --- Step 3: Format and Upload Embeddings ---
        embeddings_gcs_uri = format_and_upload_embeddings(
            embeddings, staging_bucket
        )

        # --- Step 4: Create Matching Engine Index ---
        # Use a timestamp to ensure the display name is unique.
        timestamp = time.strftime("%Y%m%d%H%M%S")
        index_display_name = f"{index_display_name_base}_{timestamp}"
        index = create_matching_engine_index(
            project_id=project_id,
            location=region,
            display_name=index_display_name,
            embeddings_gcs_uri=embeddings_gcs_uri,
        )

        # --- Step 5: Create Index Endpoint ---
        endpoint_display_name = f"{endpoint_display_name_base}_{timestamp}"
        endpoint = create_index_endpoint(
            project_id=project_id,
            location=region,
            display_name=endpoint_display_name,
        )

        # --- Step 6: Deploy Index to Endpoint ---
        logger.info(f"Deploying index '{index.display_name}' to endpoint '{endpoint.display_name}'")
        deployed_index_id = f"{DEPLOYED_INDEX_ID_PREFIX}_{timestamp}"
        endpoint.deploy_index(
            index=index,
            deployed_index_id=deployed_index_id,
        )

        logger.info("---" * 10)
        logger.info("✅ Vector Store creation and deployment complete!")
        logger.info(f"   - Index ID: {index.name}")
        logger.info(f"   - Index Endpoint ID: {endpoint.name}")
        logger.info(f"   - Deployed Index ID: {deployed_index_id}")
        logger.info(f"   - Public Endpoint Domain: {endpoint.public_endpoint_domain_name}")
        logger.info("---" * 10)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Vertex AI Vector Store from a GCS bucket.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--project_id",
        type=str,
        help="Your Google Cloud project ID. Overrides the $PROJECT_ID environment variable.",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="The Google Cloud region for your resources (e.g., 'us-central1'). Overrides the $REGION environment variable.",
    )
    parser.add_argument(
        "--source_gcs_bucket",
        type=str,
        help="The name of the GCS bucket containing the source text files (without 'gs://'). Overrides $SOURCE_GCS_BUCKET.",
    )
    parser.add_argument(
        "--staging_gcs_bucket",
        type=str,
        help="The name of the GCS bucket for staging embeddings and storing the index (without 'gs://'). Overrides $STAGING_GCS_BUCKET.",
    )
    parser.add_argument(
        "--index_display_name",
        type=str,
        help="A display name for the new Matching Engine Index. Overrides $INDEX_DISPLAY_NAME.",
    )
    parser.add_argument(
        "--index_endpoint_display_name",
        type=str,
        help="A display name for the new Index Endpoint. Overrides $INDEX_ENDPOINT_DISPLAY_NAME.",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        help="The name of the Vertex AI embedding model to use. Overrides $EMBEDDING_MODEL_NAME.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)