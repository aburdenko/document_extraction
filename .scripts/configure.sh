#! /usr/bin/bash
# Run with source /home/user/cvs-specialty-pharma/.scripts/configure.sh


# Create the virtual environment
# Check if .venv directory already exists to avoid errors on re-runs
if [ ! -d ".venv" ]; then
  # Update package lists and install python3-venv if needed
  sudo apt update && sudo apt install -y python3-venv
  echo "Creating Python virtual environment '.venv'..."
  python3 -m venv .venv
else
  echo "Virtual environment '.venv' already exists."
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pip install -r $SCRIPT_DIR/requirements.txt


# --- Authentication Setup ---
# Define the expected path to the service account key file
# Assuming it's in the same directory as the script for this example
#SERVICE_ACCOUNT_KEY_FILE="${SCRIPT_DIR}/../.creds/service_account.json" # Or provide an absolute path
SERVICE_ACCOUNT_KEY_FILE="${SCRIPT_DIR}/.creds/service_account.json" # Or provide an absolute path

# Check if the service account key file exists
if [ -f "$SERVICE_ACCOUNT_KEY_FILE" ]; then
  echo "Setting GOOGLE_APPLICATION_CREDENTIALS to use '$SERVICE_ACCOUNT_KEY_FILE'"
  # Export the variable so it's available to subsequent processes
  # Note: This export will only last for the duration of this script's execution
  # and any child processes it starts. It won't persist in the user's terminal session.
  export GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_KEY_FILE"
  echo $GOOGLE_APPLICATION_CREDENTIALS

  # You might want to activate the venv and install dependencies here if needed
  # source .venv/bin/activate
  # pip install -r requirements.txt # Example

else
  echo "Error: Service account key file not found at '$SERVICE_ACCOUNT_KEY_FILE'"
  echo "Please place the key file named 'service_account.json' in the script directory or update the path in configure.sh."
  # Exit the script if the key file is essential for configuration
  # exit 1
fi

echo "Configuration script finished."
echo "Remember to activate the virtual environment in your terminal before running Python scripts:"
echo "source .venv/bin/activate"
echo "And ensure GOOGLE_APPLICATION_CREDENTIALS is set if running outside this script's context."
