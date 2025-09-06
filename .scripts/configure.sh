# Usage: source .scripts/configure.sh
git config --global user.email "aburdenko@yahoo.com"
git config --global user.name "Alex Burdenko"

# Get the absolute path of the directory containing this script.
SCRIPT_DIR_CONFIGURE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Google Credentials Setup ---
# The service account key file should be in the root of the project directory.
SERVICE_ACCOUNT_KEY_FILE="$SCRIPT_DIR_CONFIGURE/../../service_account.json"

if [ ! -f "$SERVICE_ACCOUNT_KEY_FILE" ]; then
  echo "Error: Service account key file not found at '$PWD/$SERVICE_ACCOUNT_KEY_FILE'" >&2
  echo "Please place 'service_account.json' in your project's root directory." >&2
  return 1
fi

# --- Project Configuration ---
# All project-wide configuration variables are set here.
# These are used by the various Python scripts in this project.
# *** FIX ***
# Dynamically determine the Project ID from the service account key to prevent mismatches.
# This ensures the project used to build resource names matches the project where
# operations are executed (determined by the credentials).
export PROJECT_ID=$(jq -r .project_id "$SERVICE_ACCOUNT_KEY_FILE")
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID # Also set this common env var for client libraries
# First, ensure your gcloud CLI is configured with your project ID
gcloud config set project $PROJECT_ID

# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# The IAM service account the Cloud Function will run as.
# This is set to match the service account used for local testing to ensure consistent permissions.
export FUNCTION_SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"

export REGION="us-central1"
export LOG_NAME="extract_pipeline_log"

# --- Document AI Configuration ---
export GCS_DOCUMENT_URI="gs://extract_pipeline_bucket" # The document to process.
export DOCAI_LOCATION="us" # The multi-region for the Document AI processor (e.g., 'us' or 'eu').
export PROCESSOR_ID="faf306856e4fe9b7"
export PROCESSOR_VERSION_ID="6d0304e3791c55fb"
#export PROCESSOR_VERSION_ID="cde-v1-2025-09-01"

# --- GCS Bucket & Docker Configuration for Pipelines ---
# IMPORTANT: Bucket names must be globally unique.
export SOURCE_GCS_BUCKET=$(echo $GCS_DOCUMENT_URI | sed 's#gs://##' | cut -d'/' -f1)
export STAGING_GCS_BUCKET="${PROJECT_ID}-staging" # Bucket for pipeline artifacts and staging files
export DOCKER_REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/pipelines-repo" # Artifact Registry repo
export GCS_OUTPUT_URI="gs://${STAGING_GCS_BUCKET}/docai-output/" # Output for batch DocAI jobs


echo "Ensuring pipeline resources exist..."
gcloud storage buckets describe gs://$STAGING_GCS_BUCKET &>/dev/null || gcloud storage buckets create gs://$STAGING_GCS_BUCKET --project=$PROJECT_ID -l $REGION
gcloud artifacts repositories describe pipelines-repo --location=us-central1 &>/dev/null || gcloud artifacts repositories create pipelines-repo --repository-format=docker --location=us-central1 --description="Docker repository for Vertex AI Pipelines"

# --- Vector Store Configuration ---
# IMPORTANT: Bucket names must be globally unique.
# Using your project ID in the bucket name is a good practice.
export INDEX_DISPLAY_NAME="extract_pipeline_bucket-store-index"
export INDEX_ENDPOINT_DISPLAY_NAME="extract_pipeline_bucket-vector-store-endpoint"
export EMBEDDING_MODEL_NAME="text-embedding-004"

# --- Virtual Environment Setup ---
if [ ! -d ".venv/python3.12" ]; then
  echo "Python virtual environment '.python3.12' not found."
  echo "Attempting to install python3-venv..."
  # Run apt-get update, but don't exit immediately on failure.
  # We capture the output to inspect it for specific, non-critical errors.
  update_output=$(sudo apt-get update 2>&1)
  update_exit_code=$?
  echo "$update_output" # Display the output to the user.

  if [ $update_exit_code -ne 0 ]; then
    # Check for the common, non-blocking "Release file" error.
    if echo "$update_output" | grep -q "does not have a Release file"; then
      echo "-------------------------------------------------------------------" >&2
      echo "WARNING: 'apt-get update' failed for a repository (e.g., 'baltocdn')." >&2
      echo "The script will attempt to continue, but you should fix the system's" >&2
      echo "repository list in '/etc/apt/sources.list.d/' for long-term stability." >&2
      echo "-------------------------------------------------------------------" >&2
    else
      # For other, more critical apt-get update errors, we stop.
      echo "-------------------------------------------------------------------" >&2
      echo "ERROR: 'sudo apt-get update' failed with a critical error." >&2
      echo "Please review the output above and resolve the system's APT issues before continuing." >&2
      echo "-------------------------------------------------------------------" >&2
      return 1 # Stop sourcing the script
    fi
  fi
  if ! sudo apt-get install -y python3.12-venv; then
    echo "-------------------------------------------------------------------" >&2
    echo "ERROR: Failed to install 'python3.12-venv'." >&2
    echo "This may be due to the 'apt-get update' issue above or other system problems." >&2
    echo "-------------------------------------------------------------------" >&2
    return 1
  fi

  echo "Creating Python virtual environment '.venv/python3.12'..."
  /usr/bin/python3 -m venv .venv/python3.12
  echo "Installing dependencies into .venv/python3.12 from requirements.txt..."

  echo "Granting Service Agent permissions on GCS buckets..."
  VERTEX_AI_SERVICE_AGENT="service-$PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com"
  DOCAI_SERVICE_AGENT="service-$PROJECT_NUMBER@gcp-sa-documentai.iam.gserviceaccount.com"

  # Grant the Vertex AI Service Agent permission to read from buckets
  # (Needed for creating Vector Search indexes from GCS)
  gcloud storage buckets add-iam-policy-binding gs://$SOURCE_GCS_BUCKET \
    --member="serviceAccount:$VERTEX_AI_SERVICE_AGENT" \
    --role="roles/storage.objectViewer"

  gcloud storage buckets add-iam-policy-binding gs://$STAGING_GCS_BUCKET \
    --member="serviceAccount:$VERTEX_AI_SERVICE_AGENT" \
    --role="roles/storage.objectViewer"

  # Grant the Document AI Service Agent permissions for batch processing
  gcloud storage buckets add-iam-policy-binding gs://$SOURCE_GCS_BUCKET --member="serviceAccount:$DOCAI_SERVICE_AGENT" --role="roles/storage.objectViewer" # Read input
  gcloud storage buckets add-iam-policy-binding gs://$STAGING_GCS_BUCKET --member="serviceAccount:$DOCAI_SERVICE_AGENT" --role="roles/storage.objectAdmin" # Write output

  # --- Ensure 'unzip' is installed for VSIX validation ---
  if ! command -v unzip &> /dev/null; then
    echo "'unzip' command not found. Attempting to install..."
    sudo apt-get update && sudo apt-get install -y unzip
  fi

  # --- Ensure 'jq' is installed for robust JSON parsing ---
  if ! command -v jq &> /dev/null; then
    echo "'jq' command not found. Attempting to install..."
    sudo apt-get update && sudo apt-get install -y jq
  fi

  # --- VS Code Extension Setup (One-time) ---
  echo "Checking for 'emeraldwalk.runonsave' VS Code extension..."
  # Use the full path to the executable, which we know from the environment
  CODE_OSS_EXEC="/opt/code-oss/bin/codeoss-cloudworkstations"

  if ! $CODE_OSS_EXEC --list-extensions | grep -q "emeraldwalk.runonsave"; then
    echo "Extension not found. Installing 'emeraldwalk.runonsave'..."

    # Using the static URL as requested. Note: This points to an older version (0.3.2)
    # and replaces the logic that dynamically finds the latest version.
    VSIX_URL="https://www.vsixhub.com/go.php?post_id=519&app_id=65a449f8-c656-4725-a000-afd74758c7e6&s=v5O4xJdDsfDYE&link=https%3A%2F%2Fmarketplace.visualstudio.com%2F_apis%2Fpublic%2Fgallery%2Fpublishers%2Femeraldwalk%2Fvsextensions%2FRunOnSave%2F0.3.2%2Fvspackage"
    VSIX_FILE="/tmp/emeraldwalk.runonsave.vsix" # Use /tmp for the download

    echo "Downloading extension from specified static URL..."
    # Use curl with -L to follow redirects and -o to specify output file
    # Add --fail to error out on HTTP failure and -A to specify a browser User-Agent
    if curl --fail -L -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" -o "$VSIX_FILE" "$VSIX_URL"; then
      echo "Download complete. Installing..."
      # Add a check to ensure the downloaded file is a valid zip archive (.vsix)
      if unzip -t "$VSIX_FILE" &> /dev/null; then
        if $CODE_OSS_EXEC --install-extension "$VSIX_FILE"; then
          echo "Extension 'emeraldwalk.runonsave' installed successfully."
          echo "IMPORTANT: Please reload the VS Code window to activate the extension."
        else
          echo "Error: Failed to install the extension from '$VSIX_FILE'." >&2
        fi
      else
        echo "Error: Downloaded file is not a valid VSIX package. It may be an HTML page." >&2
        echo "Please check the VSIX_URL in the script or your network connection." >&2
      fi
      # Clean up the downloaded file
      rm -f "$VSIX_FILE" # This will run regardless of install success/failure
    else
      echo "Error: Failed to download the extension from '$VSIX_URL'." >&2
    fi
  else
    echo "Extension 'emeraldwalk.runonsave' is already installed."
  fi
else
  echo "Virtual environment '.python3.12' already exists."
fi

if type deactivate &>/dev/null; then
  echo "Deactivating existing virtual environment..."
  deactivate
fi

echo "Activating environment './venv/python3.12'..."
 . .venv/python3.12/bin/activate

# Ensure dependencies are installed/updated every time the script is sourced.
# This prevents ModuleNotFoundError if requirements.txt changes after the
# virtual environment has been created.
echo "Ensuring dependencies from requirements.txt are installed..."
 # Use the full path to the venv pip to ensure we're installing in the correct environment.
./.venv/python3.12/bin/pip install -r requirements.txt > /dev/null

echo "Service account key found. Exporting GOOGLE_APPLICATION_CREDENTIALS."
export GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_KEY_FILE"

# --- Create .env file for python-dotenv ---
# This allows local development tools (like the functions-framework) to load
# environment variables without needing to source this script every time.
ENV_FILE=".env"
echo "Creating/updating ${ENV_FILE} for local development..."

# Use a temporary file to avoid issues, then move it into place.
TEMP_ENV_FILE=$(mktemp)

{
  echo "PROJECT_ID=${PROJECT_ID}"
  echo "GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}"
  echo "FUNCTION_SERVICE_ACCOUNT=${FUNCTION_SERVICE_ACCOUNT}"
  echo "REGION=${REGION}"
  echo "DOCAI_LOCATION=${DOCAI_LOCATION}"
  echo "PROCESSOR_ID=${PROCESSOR_ID}"
  echo "PROCESSOR_VERSION_ID=${PROCESSOR_VERSION_ID}"
  echo "LOG_NAME=${LOG_NAME}"
  echo "DRIVE_SHARE_EMAIL=${DRIVE_SHARE_EMAIL}"
  echo "GEMINI_MODEL_NAME=${GEMINI_MODEL_NAME}"
  echo "JUDGEMENT_MODEL_NAME=${JUDGEMENT_MODEL_NAME}"
  echo "EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}"
  echo "SOURCE_GCS_BUCKET=${SOURCE_GCS_BUCKET}"
  echo "GCS_OUTPUT_URI=${GCS_OUTPUT_URI}"
  echo "STAGING_GCS_BUCKET=${STAGING_GCS_BUCKET}"
  echo "DOCKER_REPO=${DOCKER_REPO}"
  echo "INDEX_DISPLAY_NAME=${INDEX_DISPLAY_NAME}"
  echo "INDEX_ENDPOINT_DISPLAY_NAME=${INDEX_ENDPOINT_DISPLAY_NAME}"
  echo "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}"
} > "$TEMP_ENV_FILE"
mv "$TEMP_ENV_FILE" "$ENV_FILE"

# This POSIX-compliant check ensures the script is sourced, not executed.
# (return 0 2>/dev/null) will succeed if sourced and fail if executed.
if ! (return 0 2>/dev/null); then
  echo "-------------------------------------------------------------------"
  echo "ERROR: This script must be sourced, not executed."
  echo "Usage: source .scripts/configure.sh"
  echo "-------------------------------------------------------------------"
  exit 1
fi
