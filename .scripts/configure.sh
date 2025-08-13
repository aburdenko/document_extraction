# Usage: source .scripts/configure.sh
git config --global user.email "aburdenko@yahoo.com"
git config --global user.name "Alex Burdenko"

# --- Virtual Environment Setup ---
if [ ! -d ".venv/python3.12" ]; then
  echo "Python virtual environment '.python3.12' not found."
  echo "Attempting to install python3-venv..."
  sudo apt update && sudo apt install -y python3-venv
  echo "Creating Python virtual environment '.venv/python3.12'..."
  /usr/bin/python3 -m venv .venv/python3.12
  echo "Installing dependencies into .venv/python3.12 from requirements.txt..."
  # Path is relative to the project root, where this script is intended to be sourced from.
  ./.venv/python3.12/bin/pip install -r .scripts/requirements.txt

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

echo "Activating environment './venv/python3.12'..."
 . .venv/python3.12/bin/activate

# Ensure dependencies are installed/updated every time the script is sourced.
# This prevents ModuleNotFoundError if requirements.txt changes after the
# virtual environment has been created.
echo "Ensuring dependencies from .scripts/requirements.txt are installed..."
pip install -r .scripts/requirements.txt > /dev/null

# --- Project Configuration ---
# All project-wide configuration variables are set here.
# These are used by the various Python scripts in this project.
export PROJECT_ID="kallogjeri-project-345114" # Your Google Cloud project ID.
export REGION="us-central1"
# Use the latest stable model versions. The previous names were incorrect or pointed to older versions.
# 'gemini-1.5-flash-latest' is the correct name for the latest flash model.
export GEMINI_MODEL_NAME="gemini-2.5-pro"
#export GEMINI_MODEL_NAME="gemini-2.5-flash"
# 'text-embedding-004' is the latest stable text embedding model, replacing the older 'textembedding-gecko@003'.
export EMBEDDING_MODEL_NAME="text-embedding-004"

# --- GitHub Mirroring Configuration ---
export GITHUB_REPO_URL="https://github.com/hcls-solutions/rcm-agents/tree/main/test_data/"
export GITHUB_REPO_BRANCH="main" # Default branch for the repository
export GITHUB_TARGET_BUCKET="agentspace_hcls_demo" # IMPORTANT: Set a globally unique bucket name.
export GITHUB_TOKEN="" # Optional: Your GitHub personal access token to increase API rate limits.

# --- Vector Store Configuration ---
# IMPORTANT: Bucket names must be globally unique.
# Using your project ID in the bucket name is a good practice.
export SOURCE_GCS_BUCKET="agentspace_hcls_demo"
export STAGING_GCS_BUCKET="agentspace_hcls_demo"
export INDEX_DISPLAY_NAME="agentspace_hcls_demo-store-index"
export INDEX_ENDPOINT_DISPLAY_NAME="agentspace_hcls_demo-vector-store-endpoint"


# --- Google Credentials Setup ---
SERVICE_ACCOUNT_KEY_FILE="$HOME/service_account.json" # Path points to the user's home directory

if [ -f "$SERVICE_ACCOUNT_KEY_FILE" ]; then
  echo "Service account key found. Exporting GOOGLE_APPLICATION_CREDENTIALS."
  export GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_KEY_FILE"
else
  echo "Error: Service account key file not found at '$SERVICE_ACCOUNT_KEY_FILE'"
  echo "Please place 'service_account.json' in your home directory ($HOME) or update the path in configure.sh."
fi

# This POSIX-compliant check ensures the script is sourced, not executed.
# (return 0 2>/dev/null) will succeed if sourced and fail if executed.
if ! (return 0 2>/dev/null); then
  echo "-------------------------------------------------------------------"
  echo "ERROR: This script must be sourced, not executed."
  echo "Usage: source .scripts/configure.sh"
  echo "-------------------------------------------------------------------"
  exit 1
fi
