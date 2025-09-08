#!/usr/bin/env python3
"""
This script implements the manual RAG pipeline:
1. Processes a document using a Document AI Custom Extractor.
2. Chunks the extracted entities into meaningful text snippets.
3. Generates vector embeddings for each chunk using a Vertex AI model.
4. Finds or creates a Vector Search Index Endpoint.
5. Deploys the corresponding Vector Search Index if not already deployed.
6. Upserts the vectors and their metadata into the index.
"""

import os
import re
import json
import uuid
import argparse
from typing import List, Dict, Any, Optional

from google.api_core.client_options import ClientOptions
from google.api_core import exceptions
from google.cloud import documentai
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account

CONFIDENCE_THRESHOLD = 0.9


def batch_process_documents_with_extractor(
    project_id: str,
    location: str,
    processor_id: str,
    gcs_input_uris: List[str],
    gcs_output_uri: str,
    processor_version_id: Optional[str] = None, # New parameter
    timeout: int = 1800,  # 30 minutes
) -> List[Dict[str, Any]]:
    """
    Processes multiple documents asynchronously using Document AI's batch processing.
    """
    print(f"Batch processing {len(gcs_input_uris)} documents...")

    # When running locally, GOOGLE_APPLICATION_CREDENTIALS is set and we use the
    # service account key to ensure the correct project is used. When running in a
    # Vertex AI Pipeline or other managed environment, the variable is not set,
    # and we rely on Application Default Credentials (ADC) which use the
    # job's service account.
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials = None
    if credentials_path:
        print(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
    else:
        print("GOOGLE_APPLICATION_CREDENTIALS not set. Using Application Default Credentials (ADC).")

    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    docai_client = documentai.DocumentProcessorServiceClient(
        credentials=credentials, client_options=client_options
    )
    storage_client = storage.Client(project=project_id, credentials=credentials)

    print(f"Cleaning up GCS output path: {gcs_output_uri}")
    match = re.match(r"gs://([^/]+)/(.+)", gcs_output_uri)
    if match:
        bucket_name, prefix = match.groups()
        if not prefix.endswith('/'):
            prefix += '/'
        bucket = storage_client.bucket(bucket_name)
        blobs_to_delete = list(bucket.list_blobs(prefix=prefix))
        if blobs_to_delete:
            print(f"Deleting {len(blobs_to_delete)} existing objects from output path...")
            bucket.delete_blobs(blobs_to_delete)
            print("Cleanup complete.")
    else:
        print("Warning: Could not parse GCS output URI to perform cleanup.")

    gcs_documents = [
        documentai.GcsDocument(gcs_uri=uri, mime_type="application/pdf")
        for uri in gcs_input_uris
    ]

    input_config = documentai.BatchDocumentsInputConfig(
        gcs_documents=documentai.GcsDocuments(documents=gcs_documents)
    )

    output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=gcs_output_uri
            # NOTE: Removing field_mask. While it seems correct, the persistent
            # InvalidArgument error suggests a possible incompatibility between this
            # parameter and the specific processor version being used.
            # Let the API use its default behavior of returning all fields.
        )
    )
    
    # *** FIX ***
    # Construct the processor name, explicitly using a version ID if provided.
    # This avoids "InvalidArgument" errors caused by missing or undeployed
    # default processor versions.
    if processor_version_id:
        processor_name = docai_client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
        print(f"Using specific processor version: {processor_version_id}")
    else:
        processor_name = docai_client.processor_path(project_id, location, processor_id)
        print("Using default processor version. If this fails, specify a --processor-version-id.")


    request = documentai.BatchProcessRequest(
        name=processor_name, # Use the correctly constructed name
        input_documents=input_config,
        document_output_config=output_config,
        skip_human_review=True,
    )

    # --- NEW: Add detailed request logging for debugging ---
    print("\n--- Submitting Document AI Batch Request ---")
    # Convert the request to a dictionary for readable printing
    try:
        request_dict = documentai.BatchProcessRequest.to_dict(request)
        # Pretty-print the JSON representation of the request
        print(json.dumps(request_dict, indent=2))
    except Exception as e:
        print(f"Could not serialize request for logging: {e}")
    print("------------------------------------------\n")
    # --- END NEW ---

    operation = docai_client.batch_process_documents(request)
    print(f"Started batch processing operation: {operation.operation.name}")
    print(f"Waiting for operation to complete (timeout: {timeout}s)...")

    try:
        operation.result(timeout=timeout)
        print("Batch processing completed successfully.")
    except Exception as e:
        print(f"ERROR: Batch processing failed: {e}")
        try:
            metadata = documentai.BatchProcessMetadata.deserialize(operation.operation.metadata.value)
            print("--- Operation Error Details ---")
            print(f"State: {metadata.state}")
            for process_status in metadata.individual_process_statuses:
                if process_status.status.code != 0:
                    print(f"  - Failed file: {process_status.input_gcs_source}")
                    print(f"    Status: {process_status.status.message} (Code: {process_status.status.code})")
            print("-----------------------------")
        except Exception as meta_e:
            print(f"Could not parse detailed operation metadata: {meta_e}")
        raise

    print("Parsing results from GCS output...")
    processed_documents = []
    metadata = documentai.BatchProcessMetadata.deserialize(operation.operation.metadata.value)

    for process in metadata.individual_process_statuses:
        if not process.output_gcs_destination:
            print(f"  - WARNING: No output for input {process.input_gcs_source}. Status: {process.status.message}")
            continue

        match = re.match(r"gs://([^/]+)/(.+)", process.output_gcs_destination)
        if not match:
            continue

        bucket_name, prefix = match.groups()
        bucket = storage_client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix):
            if ".json" in blob.name.lower():
                print(f"  - Parsing result from: gs://{bucket_name}/{blob.name}")
                json_string = blob.download_as_bytes().decode("utf-8")
                document = documentai.Document.from_json(json_string, ignore_unknown_fields=True)
                processed_documents.append({"source_uri": process.input_gcs_source, "document": document})
                break

    print(f"Successfully parsed {len(processed_documents)} documents.")
    return processed_documents

def create_chunks_from_entities(document: documentai.Document, source_uri: str) -> List[Dict[str, Any]]:
    """Creates context-rich text chunks from high-confidence extracted entities."""
    print("Creating text chunks from entities...")
    chunks = []
    for entity in document.entities:
        if entity.confidence < CONFIDENCE_THRESHOLD:
            continue

        text_chunk = f"{entity.type_.replace('_', ' ').title()}: {entity.mention_text}"

        chunk_data = {
            "id": str(uuid.uuid4()),
            "text_chunk": text_chunk,
            "metadata": {
                "source_document": source_uri,
                "entity_type": entity.type_,
                "confidence": entity.confidence,
                "page_number": int(entity.page_anchor.page_refs[0].page) if entity.page_anchor.page_refs else 0,
            }
        }
        chunks.append(chunk_data)
        
    print(f"Created {len(chunks)} high-confidence chunks.")
    return chunks

def embed_chunks(project_id: str, region: str, model_name: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates embeddings for a list of text chunks using a Vertex AI embedding model."""
    print(f"Generating embeddings for {len(chunks)} chunks using model '{model_name}'...")
    aiplatform.init(project=project_id, location=region)
    
    model = TextEmbeddingModel.from_pretrained(model_name)
    text_contents = [c["text_chunk"] for c in chunks]
    
    embeddings = []
    for i in range(0, len(text_contents), 250):
        batch = text_contents[i:i + 250]
        response = model.get_embeddings(batch)
        embeddings.extend([prediction.values for prediction in response])
        print(f"Embedded batch {i//250 + 1}...")
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
        
    print("Embeddings generated successfully.")
    return chunks

def get_gcs_files_to_process(project_id: str, gcs_uri: str) -> List[str]:
    """Lists all processable PDF files from a GCS URI."""
    print(f"Listing files to process from: {gcs_uri}")
    storage_client = storage.Client(project=project_id)

    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with 'gs://'")
    
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    files_to_process = []
    bucket = storage_client.bucket(bucket_name)
    
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.lower().endswith(".pdf"):
            files_to_process.append(f"gs://{bucket_name}/{blob.name}")
            print(f"  - Found PDF: gs://{bucket_name}/{blob.name}")
        elif not blob.name.endswith('/'):
            print(f"  - Skipping non-PDF file: {blob.name}")

    if not files_to_process:
        print(f"WARNING: No PDF files found at '{gcs_uri}'.")

    return files_to_process

def upsert_to_vector_search(
    project_id: str,
    region: str,
    index_display_name: str,
    index_endpoint_display_name: str,
    vectorized_chunks: List[Dict[str, Any]],
):
    """Upserts vectorized data to a Vector Search index."""
    aiplatform.init(project=project_id, location=region)

    print(f"Checking for existing index with display name: '{index_display_name}'")
    indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')
    if not indexes:
        print(f"ERROR: No Vector Search index found with display name '{index_display_name}'.")
        print("Please run '.scripts/create_vector_store_index.py' first.")
        exit(1)
    index = indexes[0]
    print(f"Found index: {index.resource_name}")

    print(f"Checking for existing index endpoint with display name: '{index_endpoint_display_name}'")
    endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{index_endpoint_display_name}"')
    if endpoints:
        index_endpoint = endpoints[0]
        print(f"Found existing index endpoint: {index_endpoint.resource_name}")
    else:
        print(f"Creating new index endpoint: '{index_endpoint_display_name}'")
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=index_endpoint_display_name, public_endpoint_enabled=True
        )
        print(f"Created index endpoint: {index_endpoint.resource_name}")

    deployed_index_id = None
    for deployed_index in index_endpoint.deployed_indexes:
        if deployed_index.index == index.resource_name:
            deployed_index_id = deployed_index.id
            print(f"Index '{index.display_name}' is already deployed with ID: {deployed_index_id}")
            break
    
    if not deployed_index_id:
        print(f"Deploying index '{index.display_name}' to endpoint '{index_endpoint.display_name}'...")
        sanitized_display_name = "".join(c if c.isalnum() else '_' for c in index.display_name)
        deployment_id = f"idx_{sanitized_display_name[:50]}"
        index_endpoint.deploy_index(index=index, deployed_index_id=deployment_id)
        print("Index deployment initiated. The script will wait for completion (this can take 30-60 mins).")
        print("Index deployed successfully.")
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint.resource_name)
        for deployed_index in index_endpoint.deployed_indexes:
            if deployed_index.index == index.resource_name:
                deployed_index_id = deployed_index.id
                break

    if not deployed_index_id:
         print("ERROR: Could not find or create a deployment for the index on the endpoint.")
         exit(1)

    print(f"Upserting {len(vectorized_chunks)} vectors to deployed index '{deployed_index_id}'...")
    datapoints = []
    for chunk in vectorized_chunks:
        datapoint = {
            "datapoint_id": chunk["id"],
            "feature_vector": chunk["embedding"],
            "restricts": [
                 {"namespace": "source_document", "allow_list": [chunk["metadata"]["source_document"]]},
                 {"namespace": "entity_type", "allow_list": [chunk["metadata"]["entity_type"]]},
            ]
        }
        datapoints.append(datapoint)

    index_endpoint.upsert_datapoints(datapoints=datapoints)
    print("Upsert complete.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the manual DocAI to Vector Search pipeline.")
    parser.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"))
    parser.add_argument("--region", type=str, default=os.getenv("REGION", "us-central1"))
    parser.add_argument("--docai-location", type=str, default=os.getenv("DOCAI_LOCATION", "us"))
    parser.add_argument("--processor-id", type=str, default=os.getenv("PROCESSOR_ID"))
    parser.add_argument("--processor-version-id", type=str, default=os.getenv("PROCESSOR_VERSION_ID"), help="Optional: Specific processor version to use (e.g., 'pretrained-foundation-model-v2.0-2023-08-29').")
    parser.add_argument("--gcs-document-uri", type=str, default=os.getenv("GCS_DOCUMENT_URI"))
    parser.add_argument("--embedding-model-name", type=str, default=os.getenv("EMBEDDING_MODEL_NAME"))
    parser.add_argument("--gcs-output-uri", type=str, default=os.getenv("GCS_OUTPUT_URI"))
    parser.add_argument("--index-display-name", type=str, default=os.getenv("INDEX_DISPLAY_NAME"))
    parser.add_argument("--index-endpoint-display-name", type=str, default=os.getenv("INDEX_ENDPOINT_DISPLAY_NAME"))
    args = parser.parse_args()

    required_vars = ["project_id", "region", "docai_location", "processor_id", "gcs_document_uri", "gcs_output_uri", "embedding_model_name", "index_display_name", "index_endpoint_display_name"]
    for var in required_vars:
        arg_name = var.replace('-', '_')
        val = getattr(args, arg_name)
        if not val or "your-" in str(val):
            print(f"ERROR: Configuration variable '{var.upper()}' is not set.")
            print("Please set it as an environment variable or pass it as a command-line argument.")
            exit(1)

    files_to_process = get_gcs_files_to_process(args.project_id, args.gcs_document_uri)

    if not files_to_process:
        print("No documents to process. Exiting.")
        exit(0)

    processed_docs = batch_process_documents_with_extractor(
        project_id=args.project_id,
        location=args.docai_location,
        processor_id=args.processor_id,
        gcs_input_uris=files_to_process,
        gcs_output_uri=args.gcs_output_uri,
        processor_version_id=args.processor_version_id, # Pass the new argument
    )

    all_chunks_to_embed = []
    for doc_info in processed_docs:
        print(f"\n--- Creating chunks for file: {doc_info['source_uri']} ---")
        chunks = create_chunks_from_entities(doc_info["document"], doc_info["source_uri"])
        all_chunks_to_embed.extend(chunks)

    if all_chunks_to_embed:
        print(f"\n--- Total of {len(all_chunks_to_embed)} chunks from {len(files_to_process)} documents to be embedded and upserted. ---")
        vectorized_chunks = embed_chunks(args.project_id, args.region, args.embedding_model_name, all_chunks_to_embed)
        
        upsert_to_vector_search(
            project_id=args.project_id,
            region=args.region,
            index_display_name=args.index_display_name,
            index_endpoint_display_name=args.index_endpoint_display_name,
            vectorized_chunks=vectorized_chunks
        )
        print("\n✅ Pipeline finished successfully!")
    else:
        print("\nNo high-confidence entities found in any of the documents to process. Pipeline finished.")