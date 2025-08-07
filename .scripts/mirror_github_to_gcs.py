#!/usr/bin/env python3
"""
This script downloads a GitHub repository or a specific subdirectory from it
and extracts its contents directly into a specified Google Cloud Storage (GCS) bucket.

It performs the following steps:
1. Reads configuration (GitHub URL, GCS Bucket Name) from environment variables
   or command-line arguments.
2. Parses the GitHub URL to identify the owner, repository, branch, and any specific subdirectory path.
3. If a subdirectory is specified, it uses the GitHub API to recursively download only the contents of that path.
4. If no subdirectory is specified, it first attempts to download the entire repository as a single zip file for efficiency.
5. If the zip download fails (e.g., with a 404 Not Found error), it falls back to the recursive API download method for the entire repository.
6. Uploads each file to the specified GCS bucket, creating the bucket if it
   does not exist.
"""

import argparse
import logging
import os
import sys
import requests
import zipfile
import io
import re

from google.cloud import storage
from google.api_core import exceptions

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_github_url(url: str) -> dict[str, str | None]:
    """Parses a GitHub URL to extract owner, repo, branch, and path."""
    # Regex to handle various GitHub URL formats, including subdirectories
    # Handles URLs like:
    # - https://github.com/owner/repo
    # - https://github.com/owner/repo.git
    # - https://github.com/owner/repo/tree/main/path/to/dir
    pattern = re.compile(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/([^/]+)(?:/(.*))?)?$")
    match = pattern.search(url)

    if not match:
        return {}

    owner, repo, branch, path = match.groups()

    # The path might be None or an empty string, normalize to empty string if None
    path = path or ""
    # Strip trailing slashes from path
    path = path.rstrip("/")

    return {"owner": owner, "repo": repo, "branch": branch, "path": path}


def download_repo_via_api(
    session: requests.Session,
    owner: str,
    repo: str,
    branch: str,
    bucket: storage.Bucket,
    path_in_repo: str = "",
    base_path_to_strip: str = "",
) -> int:
    """
    Recursively downloads repository contents via the GitHub API and uploads to GCS.
    Returns the number of files uploaded.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_in_repo}"
    params = {"ref": branch}

    logger.info(f"  - Querying API for contents of: '{path_in_repo if path_in_repo else '/'}' on branch '{branch}'")

    try:
        response = session.get(api_url, params=params, timeout=60)
        response.raise_for_status()
        contents = response.json()

        # If the path points to a single file, the response is a dict, not a list.
        if isinstance(contents, dict) and contents.get("type") == "file":
            contents = [contents]

        upload_count = 0
        for item in contents:
            if item["type"] == "file":
                logger.info(f"    - Downloading file: {item['path']}")
                file_response = session.get(item["download_url"], timeout=300)
                file_response.raise_for_status()

                destination_blob_name = item["path"]
                if base_path_to_strip and destination_blob_name.startswith(base_path_to_strip):
                    destination_blob_name = destination_blob_name[len(base_path_to_strip) :].lstrip("/")

                if not destination_blob_name:
                    continue

                blob = bucket.blob(destination_blob_name)
                blob.upload_from_string(file_response.content)
                upload_count += 1
            elif item["type"] == "dir":
                # Recursively download contents of the directory
                upload_count += download_repo_via_api(
                    session, owner, repo, branch, bucket, item["path"], base_path_to_strip
                )
        return upload_count
    except requests.exceptions.HTTPError as e:
        # If a sub-folder is not found, just log it and continue.
        if e.response.status_code == 404:
            logger.warning(f"Could not find contents at path '{path_in_repo}'. It might be an empty directory or submodule. Skipping.")
            return 0
        # Re-raise other HTTP errors to be caught by the main handler
        raise e

def main(args):
    """Main execution function."""
    try:
        # --- Configuration Setup ---
        github_repo_url = (args.github_repo_url or os.getenv("GITHUB_REPO_URL", "")).strip()
        target_bucket_name = args.target_gcs_bucket or os.getenv("GITHUB_TARGET_BUCKET")
        # Branch from args/env is a fallback if not in URL
        github_repo_branch_fallback = args.github_repo_branch or os.getenv("GITHUB_REPO_BRANCH", "main")
        project_id = args.project_id or os.getenv("PROJECT_ID")
        github_token = os.getenv("GITHUB_TOKEN")

        if not project_id or "your-gcp-project-id-here" in project_id:
            logger.error("Project ID is not set. Please provide it via the --project_id flag or by editing .scripts/configure.sh")
            sys.exit(1)

        if not github_repo_url:
            logger.error("GitHub repository URL is not set. Please provide it via the --github_repo_url flag or by setting $GITHUB_REPO_URL.")
            sys.exit(1)

        if not target_bucket_name or "your-github-mirror-bucket-here" in target_bucket_name:
            logger.error("Target GCS bucket is not set. Please provide it via the --target_gcs_bucket flag or by editing .scripts/configure.sh")
            sys.exit(1)

        # --- Parse GitHub URL ---
        parsed_url = parse_github_url(github_repo_url)
        if not parsed_url.get("owner") or not parsed_url.get("repo"):
            logger.error(f"Could not parse owner and repo from URL: {github_repo_url}")
            sys.exit(1)

        owner = parsed_url["owner"]
        repo = parsed_url["repo"]
        branch = parsed_url.get("branch") or github_repo_branch_fallback
        path_to_mirror = parsed_url.get("path") or ""

        # --- GCS Client Setup ---
        logger.info(f"Connecting to GCS and preparing to upload to bucket: gs://{target_bucket_name}")
        storage_client = storage.Client(project=project_id)
        try:
            bucket = storage_client.get_bucket(target_bucket_name)
        except exceptions.NotFound:
            logger.info(f"Bucket '{target_bucket_name}' not found. Creating it now...")
            bucket = storage_client.create_bucket(target_bucket_name)
            logger.info(f"Bucket '{target_bucket_name}' created.")

        # --- Download Logic ---
        upload_count = 0
        # Use a single requests session
        with requests.Session() as session:
            session.headers.update({"User-Agent": "Gemini-Code-Assist-Mirror-Script"})
            if github_token:
                logger.info("    Found GITHUB_TOKEN. Using it for authentication.")
                session.headers.update({"Authorization": f"token {github_token}"})

            # If a path is specified, go directly to API download.
            # Otherwise, attempt zipball download first.
            if path_to_mirror:
                logger.info(f"Subdirectory specified. Using API-based download for path: '{path_to_mirror}'")
                upload_count = download_repo_via_api(
                    session, owner, repo, branch, bucket, path_in_repo=path_to_mirror, base_path_to_strip=path_to_mirror
                )
            else:
                # Attempt to download the GitHub repository zip file
                zipball_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
                logger.info(f"Attempting to download repository as a zip file from: {zipball_url}")
                try:
                    response = session.get(zipball_url, stream=True, timeout=300)
                    response.raise_for_status()

                    zip_bytes_io = io.BytesIO(response.content)
                    logger.info("Repository downloaded successfully as a zip file.")

                    # Extract and upload to GCS from Zip
                    if not zipfile.is_zipfile(zip_bytes_io):
                        raise ValueError("Downloaded content is not a valid zip file.")

                    zip_bytes_io.seek(0)
                    with zipfile.ZipFile(zip_bytes_io, "r") as zip_ref:
                        file_list = zip_ref.namelist()
                        if not file_list:
                            logger.warning("Zip file is empty. Nothing to upload.")
                        else:
                            root_dir = os.path.commonprefix(file_list)
                            logger.info(f"Common root directory in zip: '{root_dir}'")
                            for member in file_list:
                                if member.endswith("/"):
                                    continue
                                file_content = zip_ref.read(member)
                                destination_blob_name = member[len(root_dir) :]
                                if not destination_blob_name:
                                    continue
                                logger.info(f"  - Uploading '{member}' to 'gs://{target_bucket_name}/{destination_blob_name}'")
                                blob = bucket.blob(destination_blob_name)
                                blob.upload_from_string(file_content)
                                upload_count += 1
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        logger.warning("Zipball download failed with 404 Not Found. Falling back to folder-by-folder download via the GitHub API.")
                        # Fallback to API-based download for the whole repo
                        upload_count = download_repo_via_api(
                            session, owner, repo, branch, bucket, path_in_repo="", base_path_to_strip=""
                        )
                    else:
                        raise e

        logger.info("---" * 10)
        logger.info(f"✅ Mirroring complete! Uploaded {upload_count} files to gs://{target_bucket_name}")
        logger.info("---" * 10)

    except requests.exceptions.RequestException as e:
        logger.error(f"A network error occurred while trying to contact GitHub: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a GitHub repository (or subdirectory) and extract it to a GCS bucket.")
    parser.add_argument("--project_id", type=str, help="Your Google Cloud project ID. Overrides $PROJECT_ID.")
    parser.add_argument("--github_repo_url", type=str, help="The URL of the GitHub repository to mirror. Overrides $GITHUB_REPO_URL.")
    parser.add_argument("--target_gcs_bucket", type=str, help="The name of the GCS bucket for the mirror. Overrides $GITHUB_TARGET_BUCKET.")
    parser.add_argument("--github_repo_branch", type=str, help="The branch to use if not specified in the URL. Overrides $GITHUB_REPO_BRANCH.")
    parsed_args = parser.parse_args()
    main(parsed_args)