#!/usr/bin/env python3
"""
This script wraps another Python script to be run as a single-step Vertex AI Pipeline.

It takes a path to a Python script as a command-line argument. It then uses the
Vertex AI SDK's `CustomJob.from_local_script` method to:
1. Package the target script.
2. Build a custom Docker container image with dependencies from `requirements.txt`.
3. Push the container image to the project's Artifact Registry.
4. Create and submit a `CustomJob` to Vertex AI to run the script in the container.

This allows any local script to be executed as a scalable, serverless job on Vertex AI
without needing to manually write Dockerfiles or pipeline definitions (KFP).
"""

import os
import sys
import uuid
import time
from google.cloud import aiplatform
from google.api_core import exceptions

def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_python_script>")
        print("This script wraps another Python script into a single-step Vertex AI Pipeline.")
        sys.exit(1)

    script_path = sys.argv[1]
    if not os.path.exists(script_path):
        print(f"Error: Script file not found at '{script_path}'")
        sys.exit(1)

    if not script_path.lower().endswith(".py"):
        print("\n❌ ERROR: The target file is not a Python script.")
        print(f"   File provided: '{os.path.basename(script_path)}'")
        print("   This task is designed to run a Python (.py) file as a Vertex AI job.")
        print("   Please open the Python script you want to run (e.g., 'update_vector_store_with_doc_ai.py') in the editor and run the task again.")
        sys.exit(1)

    # --- Configuration from Environment ---
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION")
    staging_bucket = os.getenv("STAGING_GCS_BUCKET")
    docker_repo = os.getenv("DOCKER_REPO")
    service_account = os.getenv("FUNCTION_SERVICE_ACCOUNT")

    if not all([project_id, region, staging_bucket, docker_repo, service_account]):
        print("Error: Required environment variables are not set.")
        print("Please ensure PROJECT_ID, REGION, STAGING_GCS_BUCKET, DOCKER_REPO, and FUNCTION_SERVICE_ACCOUNT are set.")
        print("This is typically done by your terminal's startup script sourcing '.scripts/configure.sh'.")
        sys.exit(1)

    # --- Pipeline and Job Naming ---
    job_id = str(uuid.uuid4())[:8]
    script_name_base = os.path.splitext(os.path.basename(script_path))[0].replace('_', '-')
    job_display_name = f"job-{script_name_base}-{job_id}"
    container_image_uri = f"{docker_repo}/{script_name_base}-runner"
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    requirements_path = os.path.join(workspace_root, "requirements.txt")

    try:
        with open(requirements_path, "r", encoding="utf-8") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Warning: requirements.txt not found at '{requirements_path}'. Proceeding without additional packages.")
        requirements = []

    print("--- Configuration ---")
    print(f"Project: {project_id}, Region: {region}")
    print(f"Staging Bucket: gs://{staging_bucket}")
    print(f"Docker Repo: {docker_repo}")
    print(f"Service Account: {service_account}")
    print(f"Script to run: {script_path}")
    print(f"Requirements: {requirements_path}")
    print(f"Job Name: {job_display_name}")
    print(f"Container Image URI: {container_image_uri}")
    print("---------------------")

    # --- NEW: Pass Environment Variables to the Vertex AI Job ---
    # The script running on Vertex AI doesn't have access to the local environment
    # variables set by `configure.sh`. We must capture them here and pass them
    # explicitly to the job's environment.
    env_vars_to_pass = {
        "PROJECT_ID": project_id,
        "REGION": region,
        "DOCAI_LOCATION": os.getenv("DOCAI_LOCATION"),
        "PROCESSOR_ID": os.getenv("PROCESSOR_ID"),
        "PROCESSOR_VERSION_ID": os.getenv("PROCESSOR_VERSION_ID"),
        "GCS_DOCUMENT_URI": os.getenv("GCS_DOCUMENT_URI"),
        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME"),
        "GCS_OUTPUT_URI": os.getenv("GCS_OUTPUT_URI"),
        "INDEX_DISPLAY_NAME": os.getenv("INDEX_DISPLAY_NAME"),
        "INDEX_ENDPOINT_DISPLAY_NAME": os.getenv("INDEX_ENDPOINT_DISPLAY_NAME"),
    }
    # Filter out any keys where the value is None to avoid sending empty strings
    job_environment = {k: v for k, v in env_vars_to_pass.items() if v is not None}
    print("\n--- Passing Environment to Job ---")
    for k, v in job_environment.items():
        print(f"  {k}: {v}")
    print("----------------------------------")
    # --- END NEW SECTION ---

    # --- Initialize Vertex AI SDK ---
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    print(f"\nPackaging '{script_path}' as a custom job...")
    job = aiplatform.CustomJob.from_local_script(
        display_name=job_display_name,
        script_path=script_path,
        container_uri=container_image_uri,
        requirements=requirements,
        # UPDATED: Pass the captured environment variables to the job at creation time.
        environment_variables=job_environment,
    )

    print("Submitting job to Vertex AI. This may take several minutes for the container build...")
    job.run(
        sync=False,
        service_account=service_account,
    )

    print("\n✅ Job submitted successfully!")
    print("You can monitor its progress in the Vertex AI 'Pipelines' or 'Training' section of the Google Cloud Console.")

    link_retrieved = False
    max_attempts = 10
    poll_interval_seconds = 3

    print("\nAttempting to retrieve job details from Vertex AI...")
    for attempt in range(max_attempts):
        # Use a more robust method to find the job by its display name.
        # This is more reliable than refreshing a single job object, especially
        # if the job fails very quickly.
        job_list = aiplatform.CustomJob.list(filter=f'display_name="{job_display_name}"', order_by="create_time desc")

        if job_list:
            latest_job = job_list[0]
            # Check for a terminal failure state.
            if latest_job.state in [
                aiplatform.job_state.JobState.JOB_STATE_FAILED,
                aiplatform.job_state.JobState.JOB_STATE_CANCELLED,
                aiplatform.job_state.JobState.JOB_STATE_EXPIRED,
            ]:
                print("\n❌ ERROR: The Vertex AI job has failed.")
                print(f"  Job State: {latest_job.state}")
                if latest_job.error:
                    print(f"  Error Message: {latest_job.error.message}")
                print("  Please check the job logs in the Google Cloud Console for the full traceback.")
                try:
                    print(f"  Direct link to failed job: {latest_job._dashboard_uri()}")
                except (RuntimeError, AttributeError):
                    print(f"  Could not retrieve direct link. Find job '{latest_job.display_name}' in the Vertex AI 'Training' > 'Custom Jobs' section.")
                link_retrieved = True
                break

            # If the job is running or succeeded, we can get the link.
            if latest_job.state in [
                aiplatform.job_state.JobState.JOB_STATE_RUNNING,
                aiplatform.job_state.JobState.JOB_STATE_SUCCEEDED,
            ]:
                print(f"  Job Resource Name: {latest_job.resource_name}")
                print(f"  Direct link to job: {latest_job._dashboard_uri()}")
                link_retrieved = True
                break
            # If the job is in another state (like PENDING or QUEUED), we just wait for the next loop.

        # If the job_list is empty after a few seconds, it likely failed to create
        # or was deleted immediately.
        if not job_list and attempt > 2: # Give it a few seconds to appear
            print("\n❌ ERROR: The Vertex AI job resource could not be found after creation.")
            print("  This usually means the job failed very quickly after starting, often due to a permissions issue.")
            print(f"  Check Cloud Logging for logs related to job display name: '{job_display_name}'")
            print("\n  Common causes for this are:")
            print("  1. The service account running the job is missing required IAM roles (e.g., Storage, Document AI, Vertex AI User).")
            print(f"  2. The service account used to submit the job needs the 'Service Account User' role on the job's service account ('{service_account}').")
            print("  3. A prerequisite is missing (e.g., the Vector Search Index does not exist or is not deployed).")
            link_retrieved = True
            break

        if attempt < max_attempts - 1 and not link_retrieved:
            print(f"  (Attempt {attempt + 1}/{max_attempts}) Waiting for job details...")
            time.sleep(poll_interval_seconds)

    if not link_retrieved:
        print("\nCould not retrieve job details after several attempts.")
        print("The job may be running, but its status could not be confirmed.")
        print("Please check the Vertex AI 'Training' > 'Custom Jobs' section in the Cloud Console.")

if __name__ == "__main__":
    main()