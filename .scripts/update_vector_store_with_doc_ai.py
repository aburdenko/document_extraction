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
    force_reprocess: bool = False,
    timeout: int = 7200,  # 2 hours
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

    # --- NEW: Check for existing results before processing ---
    if not force_reprocess:
        print(f"Checking for existing processed JSONs in {gcs_output_uri}...")
        match = re.match(r"gs://([^/]+)/(.+)", gcs_output_uri)
        if match:
            bucket_name, prefix = match.groups()
            if not prefix.endswith('/'):
                prefix += '/'
            bucket = storage_client.bucket(bucket_name)

            # Find all JSON files, but exclude our own checkpoint files.
            existing_json_blobs = [
                b for b in bucket.list_blobs(prefix=prefix)
                if b.name.lower().endswith(".json") and '.checkpoint_' not in b.name
            ]

            if existing_json_blobs:
                print(f"Found {len(existing_json_blobs)} existing JSON files. Attempting to parse and reuse.")
                parsed_docs = []
                found_uris = set()

                for blob in existing_json_blobs:
                    try:
                        json_string = blob.download_as_bytes().decode("utf-8")
                        doc = documentai.Document.from_json(json_string, ignore_unknown_fields=True)
                        if doc.uri:  # The URI field in the Document JSON links it to the source
                            parsed_docs.append({"source_uri": doc.uri, "document": doc})
                            found_uris.add(doc.uri)
                    except Exception as e:
                        print(f"  - Warning: Could not parse or read source URI from gs://{bucket_name}/{blob.name}. Error: {e}")

                required_uris = set(gcs_input_uris)
                if required_uris.issubset(found_uris):
                    print("All required documents found in existing output. Skipping Document AI batch processing.")
                    # Filter to return only the docs we were asked for, in the correct order.
                    doc_map = {d['source_uri']: d['document'] for d in parsed_docs}
                    final_docs = [{'source_uri': uri, 'document': doc_map[uri]} for uri in gcs_input_uris if uri in doc_map]
                    print(f"Successfully reused {len(final_docs)} parsed documents.")
                    return final_docs
                else:
                    print(f"Found some results, but {len(required_uris - found_uris)} are missing. Re-processing all documents.")

    if force_reprocess:
        print(f"Force reprocess enabled. Cleaning up GCS output path: {gcs_output_uri}")
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
    print(f"Waiting for operation to complete (timeout: {timeout}s)... This can take a long time for many/large files.")

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
    """
    Creates context-rich text chunks from high-confidence extracted entities,
    embedding key metadata directly into the text chunk for RAG retrieval.
    """
    print("Creating text chunks from entities with embedded metadata...")
    chunks = []
    for entity in document.entities:
        if entity.confidence < CONFIDENCE_THRESHOLD:
            continue

        # Page numbers from Document AI are 0-indexed, so add 1 for human readability.
        page_number = int(entity.page_anchor.page_refs[0].page) + 1 if entity.page_anchor.page_refs else 0
        source_filename = os.path.basename(source_uri)

        # --- Create a metadata block to embed in the text ---
        metadata_block = (
            f"[METADATA]\n"
            f"source_document: {source_filename}\n"
            f"page_number: {page_number}\n"
            f"[/METADATA]\n"
        )

        original_content = f"{entity.type_.replace('_', ' ').title()}: {entity.mention_text}"
        
        # --- Combine metadata block with original content ---
        # This makes the metadata available to the LLM at retrieval time.
        text_chunk = f"{metadata_block}{original_content}"

        chunk_data = {
            "id": str(uuid.uuid4()),
            "text_chunk": text_chunk, # This now contains the metadata
            "metadata": {
                "source_document": source_uri, # Keep full URI for filtering in Vector Search
                "entity_type": entity.type_,
                "confidence": entity.confidence,
                "page_number": page_number,
            }
        }
        chunks.append(chunk_data)
        
    print(f"Created {len(chunks)} high-confidence chunks with embedded metadata.")
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
        # The .list() method returns a summary object. To get the full object with
        # all methods, we need to re-instantiate the class using the full resource name.
        endpoint_summary = endpoints[0]
        print(f"Found existing index endpoint: {endpoint_summary.resource_name}")
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_summary.resource_name)
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
        # Deployed index IDs must start with a letter and contain only letters, numbers, and underscores.
        sanitized_display_name = re.sub(r'[^a-zA-Z0-9_]', '_', index.display_name)
        deployment_id = f"d_{sanitized_display_name[:55]}"
        index_endpoint.deploy_index(index=index, deployed_index_id=deployment_id)
        print("Index deployment initiated. The script will wait for completion (this can take 30-60 mins).")
        # The SDK call blocks until deployment is complete.
        print("Index deployed successfully.")
        # Refresh the endpoint object to see the new deployment
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

    # ============================ FIX START ============================
    # The upsert method must be called on the `index` object, not the `index_endpoint`.
    # Robustly call the upsert method on the INDEX object to handle different SDK versions.
    if hasattr(index, "upsert_datapoints"):
        index.upsert_datapoints(datapoints=datapoints)
    elif hasattr(index, "upsert"):
        # Older SDK versions used `upsert`
        index.upsert(datapoints=datapoints)
    else:
        # This case should not be reached with a valid Index object.
        print(f"FATAL: Could not find 'upsert' or 'upsert_datapoints' on object of type {type(index)}.")
        exit(1)
    # ============================= FIX END =============================

    print("Upsert complete.")

def save_chunks_as_text_files_to_gcs(
    project_id: str,
    gcs_output_uri_for_rag: str,
    processed_docs: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
):
    """
    Saves processed text chunks to GCS as .txt files, one per original document.
    This prepares the data for ingestion by the managed RAG Engine.
    """
    print(f"Saving processed text files for RAG Engine to {gcs_output_uri_for_rag}...")
    storage_client = storage.Client(project=project_id)
    match = re.match(r"gs://([^/]+)/(.*)", gcs_output_uri_for_rag)
    if not match:
        print(f"ERROR: Invalid GCS URI format: {gcs_output_uri_for_rag}")
        return

    bucket_name, prefix = match.groups()
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    bucket = storage_client.bucket(bucket_name)

    # Group chunks by their original source document
    chunks_by_source = {}
    for chunk in all_chunks:
        source_uri = chunk['metadata']['source_document']
        if source_uri not in chunks_by_source:
            chunks_by_source[source_uri] = []
        chunks_by_source[source_uri].append(chunk['text_chunk'])

    # Create a map of source_uri to full document text from the processed docs
    doc_text_by_source = {
        doc_info['source_uri']: doc_info['document'].text
        for doc_info in processed_docs
    }

    # For each source document, create a single .txt file containing the entity
    # summary and the full document text.
    for source_uri in doc_text_by_source:
        source_filename = os.path.basename(source_uri)
        base_filename, _ = os.path.splitext(source_filename)
        output_filename = f"{base_filename}.txt"
        output_blob_name = f"{prefix}{output_filename}"

        # Get the entity chunks for this document, if any
        entity_chunks = chunks_by_source.get(source_uri, [])
        entities_summary = "\n\n---\n\n".join(entity_chunks)
        full_doc_text = doc_text_by_source.get(source_uri, "--- ERROR: Full document text not found. ---")

        # Combine the extracted entities summary and the full document text.
        # This provides both a high-level summary and the complete context for the RAG engine.
        full_text_content = (
            f"[EXTRACTED_ENTITIES_SUMMARY]\n{entities_summary}\n[/EXTRACTED_ENTITIES_SUMMARY]\n\n"
            f"--- DOCUMENT CONTENT ---\n\n{full_doc_text}"
        )
        blob = bucket.blob(output_blob_name)
        blob.upload_from_string(full_text_content, content_type="text/plain; charset=utf-8")
        print(f"  - Saved gs://{bucket_name}/{output_blob_name}")

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
    parser.add_argument("--output-mode", type=str, choices=['vector-search', 'rag-engine-text-files'], default='vector-search', help="Specify the output target.")
    parser.add_argument("--gcs-rag-text-uri", type=str, default=os.getenv("GCS_RAG_TEXT_URI"), help="GCS path to save text files for RAG Engine.")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of all documents, ignoring any checkpoints.")
    parser.add_argument("--docai-timeout", type=int, default=os.getenv("DOCAI_TIMEOUT", 7200), help="Timeout in seconds for the Document AI batch processing job. Defaults to 2 hours.")
    args = parser.parse_args()

    required_vars = ["project_id", "region", "docai_location", "processor_id", "gcs_document_uri", "gcs_output_uri", "embedding_model_name", "index_display_name", "index_endpoint_display_name"]
    if args.output_mode == 'rag-engine-text-files':
        required_vars.append("gcs_rag_text_uri")

    for var in required_vars:
        # Command-line args use '-', env vars use '_'. Standardize to '_' for getattr.
        arg_name = var.replace('-', '_')
        val = getattr(args, arg_name)
        if not val or "your-" in str(val):
            print(f"ERROR: Configuration variable '{var.upper()}' is not set.")
            print("Please set it as an environment variable or pass it as a command-line argument.")
            exit(1)

    # --- Checkpoint Setup ---
    storage_client = storage.Client(project=args.project_id)
    match = re.match(r"gs://([^/]+)/(.+)", args.gcs_output_uri)
    if not match:
        print(f"ERROR: Could not parse GCS output URI for checkpointing: {args.gcs_output_uri}")
        exit(1)
    bucket_name, prefix = match.groups()
    if not prefix.endswith('/'):
        prefix += '/'
    bucket = storage_client.bucket(bucket_name)
    vector_checkpoint_blob_name = f"{prefix.strip('/')}/.checkpoint_vectorized_chunks.json"
    vector_checkpoint_blob = bucket.blob(vector_checkpoint_blob_name)

    if args.output_mode == 'vector-search' and args.force_reprocess and vector_checkpoint_blob.exists():
        print("Force reprocess requested. Deleting existing vector checkpoint file...")
        vector_checkpoint_blob.delete()
        print("Checkpoint deleted.")

    vectorized_chunks = None
    # --- Checkpoint Restore ---
    if args.output_mode == 'vector-search' and not args.force_reprocess and vector_checkpoint_blob.exists():
        print(f"Found checkpoint file at gs://{bucket_name}/{vector_checkpoint_blob_name}. Resuming from embedding step.")
        try:
            data = vector_checkpoint_blob.download_as_bytes()
            vectorized_chunks = json.loads(data)
            print(f"Successfully loaded {len(vectorized_chunks)} vectorized chunks from checkpoint.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint file. Will re-process from start. Error: {e}")
            vectorized_chunks = None

    if vectorized_chunks is None:
        files_to_process = get_gcs_files_to_process(args.project_id, args.gcs_document_uri)

        if not files_to_process:
            print("No documents to process. Exiting.")
            exit(0)

        # --- MODIFIED: Process files individually to isolate errors ---
        print("\nProcessing documents one by one to better isolate potential errors.")
        processed_docs = []
        failed_files = []

        # If force_reprocess is on, clean the ENTIRE output directory ONCE before starting.
        # This prevents each iteration from deleting the previous one's output.
        if args.force_reprocess:
            print(f"Initial cleanup of GCS output path due to --force-reprocess: {args.gcs_output_uri}")
            match = re.match(r"gs://([^/]+)/(.+)", args.gcs_output_uri)
            if match:
                bucket_name, prefix = match.groups()
                if not prefix.endswith('/'):
                    prefix += '/'
                bucket = storage.Client(project=args.project_id).bucket(bucket_name)
                blobs_to_delete = list(bucket.list_blobs(prefix=prefix))
                if blobs_to_delete:
                    print(f"Deleting {len(blobs_to_delete)} existing objects from output path...")
                    bucket.delete_blobs(blobs_to_delete)
                    print("Initial cleanup complete.")

        for file_uri in files_to_process:
            print(f"\n{'='*20} Processing: {os.path.basename(file_uri)} {'='*20}")
            try:
                # We call the batch function with a single file.
                # We set force_reprocess=False because we've already cleaned the directory.
                # The function's internal caching will now correctly check if this specific file
                # has already been processed and reuse it if possible.
                processed_docs_single = batch_process_documents_with_extractor(
                    project_id=args.project_id,
                    location=args.docai_location,
                    processor_id=args.processor_id,
                    gcs_input_uris=[file_uri],
                    gcs_output_uri=args.gcs_output_uri,
                    processor_version_id=args.processor_version_id,
                    force_reprocess=False, # Cleanup is handled outside the loop
                    timeout=args.docai_timeout,
                )
                if processed_docs_single:
                    processed_docs.extend(processed_docs_single)
                else:
                    print(f"Warning: No processed document returned for {file_uri}, but no exception was raised.")
                    failed_files.append(file_uri)

            except Exception as e:
                print(f"\n❌❌❌ FAILED to process file: {file_uri} ❌❌❌")
                print(f"Error: {e}")
                failed_files.append(file_uri)

        all_chunks = []
        for doc_info in processed_docs:
            print(f"\n--- Creating chunks for file: {doc_info['source_uri']} ---")
            chunks = create_chunks_from_entities(doc_info["document"], doc_info["source_uri"])
            all_chunks.extend(chunks)

        if not all_chunks:
            print("\nNo high-confidence entities found in any of the documents to process. Pipeline finished.")
            exit(0)

        # --- Branch logic based on the desired output ---
        if args.output_mode == 'rag-engine-text-files':
            save_chunks_as_text_files_to_gcs(
                project_id=args.project_id,
                gcs_output_uri_for_rag=args.gcs_rag_text_uri,
                processed_docs=processed_docs,
                all_chunks=all_chunks,
            )
            print("\n✅ Pipeline finished successfully! Text files for RAG Engine have been created.")
            print(f"You can now run 'create_vector_store_with_rag_engine.py', which will use the files in '{args.gcs_rag_text_uri}'.")
            exit(0) # Exit after creating text files, no further steps needed.

        elif args.output_mode == 'vector-search':
            print(f"\n--- Total of {len(all_chunks)} chunks from {len(files_to_process)} documents to be embedded and upserted. ---")
            vectorized_chunks = embed_chunks(args.project_id, args.region, args.embedding_model_name, all_chunks)

            # --- Checkpoint Save (only for vector-search mode) ---
            print(f"Checkpointing {len(vectorized_chunks)} vectorized chunks before upsert...")
            try:
                vector_checkpoint_blob.upload_from_string(json.dumps(vectorized_chunks), content_type="application/json")
                print(f"Checkpoint saved to gs://{bucket_name}/{vector_checkpoint_blob_name}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint file. Error: {e}")

    # --- Final Step: Upsert to Vector Search (if in vector-search mode) ---
    # This block is reached either by generating new vectors or by loading from a checkpoint.
    if args.output_mode == 'vector-search':
        if vectorized_chunks: # Ensure we have data to upsert
            upsert_to_vector_search(
                project_id=args.project_id,
                region=args.region,
                index_display_name=args.index_display_name,
                index_endpoint_display_name=args.index_endpoint_display_name,
                vectorized_chunks=vectorized_chunks
            )
            if vector_checkpoint_blob.exists():
                print("Upsert successful. Deleting checkpoint file...")
                vector_checkpoint_blob.delete()
                print("Checkpoint file deleted.")
            print("\n✅ Pipeline finished successfully! Data upserted to Vector Search.")
        else:
            print("\nNo high-confidence entities found in any of the documents to process. Pipeline finished.")