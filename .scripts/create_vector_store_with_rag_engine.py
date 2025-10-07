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
import re
import sys
import os
from dotenv import load_dotenv
from google.cloud import storage


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
    # Load environment variables from a .env file if it exists.
    # This is useful for local development.
    load_dotenv()

    try:
        # --- Configuration Setup ---
        project_id = args.project_id or os.getenv("PROJECT_ID")
        primary_region = args.region or os.getenv("REGION", "us-central1")
        # The script now exclusively uses a full GCS URI for its source.
        source_gcs_uri = args.source_gcs_uri or os.getenv("GCS_RAG_TEXT_URI")
        corpus_display_name = args.corpus_display_name or os.getenv("INDEX_DISPLAY_NAME", "my-rag-corpus")
        recreate = args.recreate

        # --- Validate Configuration ---
        if not project_id or "your-gcp-project-id-here" in project_id:
            logger.error("Project ID is not set. Please provide it via the --project_id flag or by editing .scripts/configure.sh")
            sys.exit(1)

        if not source_gcs_uri:
            logger.error("Source GCS URI is not set. Please provide it via the --source_gcs_uri flag or by setting GCS_RAG_TEXT_URI in .scripts/configure.sh")
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
                    rag.delete_corpus(name=corpus.name)
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
        logger.info(f"Listing files in GCS path '{source_gcs_uri}'...")
        storage_client = storage.Client(project=project_id)

        # Parse the GCS URI to get bucket and prefix
        match = re.match(r"gs://([^/]+)/(.*)", source_gcs_uri)
        if not match:
            logger.error(f"Invalid GCS URI format: {source_gcs_uri}")
            sys.exit(1)

        bucket_name, prefix = match.groups()
        bucket = storage_client.bucket(bucket_name)
        
        supported_files = []
        
        # Iterate through all blobs (files) in the bucket with the specified prefix
        for blob in bucket.list_blobs(prefix=prefix):
            if any(blob.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                supported_files.append(f"gs://{bucket_name}/{blob.name}")
                logger.info(f"  - Found supported file: {blob.name}")
            elif not blob.name.endswith('/'): # Don't log directories
                logger.info(f"  - Skipping unsupported file: {blob.name}")

        if not supported_files:
            logger.error(f"No supported files found in GCS path '{source_gcs_uri}'. Supported types: {SUPPORTED_EXTENSIONS}")
            sys.exit(1)
            
        logger.info(f"Found {len(supported_files)} supported files to import.")

        # 3. Import Filtered Files into Corpus using batches
        logger.info(f"Starting file import into corpus '{rag_corpus.name}'...")
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
                logger.info(f"Import API call for batch {int(i/batch_size) + 1} completed.")
                # Provide detailed feedback based on the API response.
                if hasattr(response, 'imported_rag_files_count') and response.imported_rag_files_count > 0:
                    logger.info(f"  - Import process started for {response.imported_rag_files_count} new/updated files.")
                if hasattr(response, 'skipped_rag_files_count') and response.skipped_rag_files_count > 0:
                    logger.warning(f"  - Skipped {response.skipped_rag_files_count} files. This is expected if they have already been imported and have not changed since the last import.")
                    logger.warning("  - If you intended to re-process all files from scratch, run this script again with the '--recreate' flag.")
                if hasattr(response, 'failed_rag_files_count') and response.failed_rag_files_count > 0:
                    logger.error(f"  - Failed to start import for {response.failed_rag_files_count} files. Check GCS permissions and file formats.")
            except google_exceptions.InvalidArgument as e:
                logger.error(f"Failed to import a batch of files: {e}", exc_info=True)
                sys.exit(1)

        logger.info("\nProcessing can take a significant amount of time depending on the number and size of files.")
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
        "--source_gcs_uri",
        type=str,
        help="The GCS URI (gs://bucket/prefix) containing the source files. Overrides $GCS_RAG_TEXT_URI.",
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