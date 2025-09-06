#!/bin/bash

# This script fetches the latest logs for the Cloud Function and saves them locally.

# Change to the project's root directory to ensure paths are correct.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit

# Source the configuration script to get PROJECT_ID, REGION, etc.
if [ -f ".scripts/configure.sh" ]; then
    source .scripts/configure.sh
else
    echo "ERROR: Configuration file .scripts/configure.sh not found." >&2
    exit 1
fi

# --- Configuration ---
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/docai_batch_logs_$(date +%Y%m%d_%H%M%S).log"
LOG_LIMIT=50 # Number of log entries to fetch

# --- Script Logic ---
echo "Fetching logs for Document AI batch operations in project: $PROJECT_ID"

# Create the log directory if it doesn't exist
mkdir -p $LOG_DIR

# Construct the filter for gcloud logging
# This filter targets the audit logs for the Document AI BatchProcessDocuments API call.
# It looks for entries with a severity of ERROR, which can provide more details
# on 'InvalidArgument' failures than the operation metadata alone.
FILTER="protoPayload.serviceName=\"documentai.googleapis.com\" AND protoPayload.methodName=\"google.cloud.documentai.v1.DocumentProcessorService.BatchProcessDocuments\" AND severity>=ERROR"

echo "Using filter: $FILTER"
echo "Fetching the last $LOG_LIMIT log entries..."

# Execute the gcloud command and save output to the file
# Using --format="json" provides the most detail, which is needed for debugging.
gcloud logging read "$FILTER" \
  --project=$PROJECT_ID \
  --limit=$LOG_LIMIT \
  --order=desc \
  --format="json" > "$LOG_FILE"

if [ $? -eq 0 ]; then
  if [ ! -s "$LOG_FILE" ]; then
    echo "✅ Success! No recent error logs found for Document AI batch processing."
    echo "Log file created (but is empty): $LOG_FILE"
  else
    echo "✅ Success! Logs have been saved to: $LOG_FILE"
    echo "You can view the logs with the command: cat $LOG_FILE"
    echo "Look for the 'status' field inside 'protoPayload' for detailed error messages."
  fi
else
  echo "❌ Error: Failed to fetch logs. Please check your gcloud authentication and permissions." >&2
  exit 1
fi