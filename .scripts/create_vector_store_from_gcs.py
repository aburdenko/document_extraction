#!/usr/bin/env python3
"""
This script creates a Vertex AI RAG Corpus from unstructured text files
stored in a Google Cloud Storage bucket. This is the preferred, modern approach
for creating a knowledge base for RAG with Gemini.

By default, the script will use an existing RAG Corpus if one with the specified
display name is found, and will only import new or updated files from the GCS
bucket. This avoids costly re-indexing of unchanged files.

Use the `--recreate` flag to force the deletion of an existing corpus and create a new one from scratch.
"""

import argparse
import logging
from google.api_core import exceptions as google_exceptions
import sys

import vertexai
from vertexai.preview import rag

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constants ---
# --- Text Chunking Configuration ---
CHUNK_SIZE = 1024  # Characters per chunk
CHUNK_OVERLAP = 200 # Characters to overlap between chunks
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.csv', '.json', '.jsonl']


def main(args):
    """Main execution function."""
    try:
        # --- Configuration Setup ---
        project_id = args.project_id or os.getenv("PROJECT_ID")
        primary_region = args.region or os.getenv("REGION", "us-central1")
        source_bucket = args.source_gcs_bucket or os.getenv("SOURCE_GCS_BUCKET")
        corpus_display_name = args.corpus_display_name or os.getenv("INDEX_DISPLAY_NAME", "my-rag-corpus")
        embedding_model_name = args.embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")
        recreate = args.recreate

        # --- Validate Configuration ---
        if not project_id or "your-gcp-project-id-here" in project_id:
            logger.error("Project ID is not set. Please provide it via the --project_id flag or by editing .scripts/configure.sh")
            sys.exit(1)

        if not source_bucket or "your-source-bucket-name-here" in source_bucket:
            logger.error("Source GCS bucket is not set. Please provide it via the --source_gcs_bucket flag or by editing .scripts/configure.sh")
            sys.exit(1)

        # --- Initialization ---
        logger.info(f"Initializing Vertex AI for project '{project_id}' in default region '{primary_region}'...")
        vertexai.init(project=project_id, location=primary_region)

        # --- GCS Import Mode ---
        logger.info("--- Running in GCS Import Mode ---")

        # 1. Find or Create RAG Corpus
        rag_corpus = None
        for corpus in rag.list_corpora():
            if corpus.display_name == corpus_display_name:
                logger.info(f"Found existing RAG Corpus with display name: '{corpus_display_name}' (Resource Name: {corpus.name})")
                if recreate:
                    logger.warning(f"Recreate flag is set. Deleting existing corpus '{corpus.name}'...")
                    corpus.delete(force=True)  # force=True to delete even if not empty
                    logger.info("Existing corpus deleted.")
                else:
                    rag_corpus = corpus
                break  # Found our match, exit loop

        if rag_corpus is None:  # Either it didn't exist or it was just deleted
            logger.info(f"Creating new RAG Corpus with display name: '{corpus_display_name}'")
            rag_corpus = rag.create_corpus(display_name=corpus_display_name)
            logger.info(f"Corpus created successfully. Resource Name: {rag_corpus.name}")
        else:
            logger.info(f"Using existing corpus '{rag_corpus.name}' for file import.")

        # 2. List and Filter Files
        logger.info(f"Listing files in GCS bucket '{source_bucket}'...")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(source_bucket)
        
        supported_files = []
        
        # Iterate through all blobs (files) in the bucket
        for blob in bucket.list_blobs(prefix=None):
            if any(blob.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                supported_files.append(f"gs://{source_bucket}/{blob.name}")
            else:
                logger.info(f"Skipping unsupported file: {blob.name}")

        if not supported_files:
            logger.error(f"No supported files found in GCS bucket 'gs://{source_bucket}'. Supported types: {SUPPORTED_EXTENSIONS}")
            sys.exit(1)
            
        logger.info(f"Found {len(supported_files)} supported files to import.")

        # 3. Import Filtered Files into Corpus using batches
        logger.info(f"Starting file import into corpus '{rag_corpus.name}'...")
        logger.info(f"Using embedding model: {embedding_model_name}")
        logger.info(f"Chunk size: {CHUNK_SIZE}, Chunk overlap: {CHUNK_OVERLAP}")

        # The API supports up to 100 files per import request.
        batch_size = 100
        for i in range(0, len(supported_files), batch_size):
            file_batch = supported_files[i:i + batch_size]
            logger.info(f"Importing batch {int(i/batch_size) + 1} of {len(supported_files)} files...")
            try:
                response = rag.import_files(
                    rag_corpus.name,
                    file_batch,  # Pass the list of supported file URIs
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                logger.info(f"Import process for batch {int(i/batch_size) + 1} started. Response: {response}")
            except google_exceptions.InvalidArgument as e:
                logger.error(f"Failed to import a batch of files: {e}", exc_info=True)
                sys.exit(1)

        logger.info("Processing can take a significant amount of time depending on the number and size of files.")
        logger.info("You can monitor the status in the Google Cloud Console under Vertex AI -> RAG Engine.")
        logger.info("---" * 10)
        logger.info("✅ RAG Corpus processing initiated successfully!")

        logger.info("---" * 10)
        logger.info(f"   Your RAG Corpus Name is: {rag_corpus.name}")
        logger.info("   Copy this full name (projects/...) and paste it into your prompt file under the '# RagEngine' section.")
        logger.info("---" * 10)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Vertex AI RAG Corpus from a GCS bucket.",
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
        "--embedding_model_name",
        type=str,
        help="The name of the Vertex AI embedding model to use. Overrides $EMBEDDING_MODEL_NAME.",
    )
    parser.add_argument(
        "--corpus_display_name",
        type=str,
        help="A display name for the new RAG Corpus. Overrides $INDEX_DISPLAY_NAME.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="If set, any existing RAG Corpus with the same display name will be deleted and a new one will be created.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)