#!/usr/bin/env python3
# Run with:
# .scripts/eval.py --all-time

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.cloud import aiplatform
import vertexai
from vertexai.preview.evaluation import EvalTask, AutoraterConfig

from google.cloud import storage
from google.cloud import logging
from datetime import datetime, timedelta
import os, argparse

# --- 1. Configuration (now using environment variables) ---
# --- These variables must be set in your Cloud Function environment ---
PROJECT_ID = os.environ.get("PROJECT_ID", "your-gcp-project-id")
LOCATION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("STAGING_GCS_BUCKET", "your-bucket-name")
# The LOG_NAME env var contains the short name of the log (e.g., 'agentspace_hcls_demo_log').
# The Cloud Logging filter requires the full resource name.
SHORT_LOG_NAME = os.environ.get("LOG_NAME", "run_gemini_from_file") # Default must match run_gemini_from_file.py
LOG_NAME = f"projects/{PROJECT_ID}/logs/{SHORT_LOG_NAME}"
JUDGEMENT_MODEL_NAME = os.environ.get("JUDGEMENT_MODEL_NAME", "gemini-2.5-flash")

# File to store the timestamp of the last run
TIMESTAMP_FILE = "last_run_timestamp.txt"

def get_last_run_timestamp():
    """Reads the timestamp of the last run from a local file."""
    try:
        with open(TIMESTAMP_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"

def save_current_timestamp():
    """Saves the current timestamp to a local file."""
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(datetime.utcnow().isoformat() + "Z")

def get_logs_for_evaluation(last_run_timestamp: str | None) -> pd.DataFrame:
    """Queries Cloud Logging for all new logs since the last run."""
    logging_client = logging.Client(project=PROJECT_ID)
    
    # Base filter to get only the structured logs with a request_id
    base_filter = (
        f'logName="{LOG_NAME}" AND '
        f'jsonPayload.request_id:*'
    )

    # Conditionally add the timestamp part of the filter
    if last_run_timestamp:
        log_filter = f'{base_filter} AND timestamp >= "{last_run_timestamp}"'
        print(f"Querying for structured logs in '{SHORT_LOG_NAME}' since: {last_run_timestamp}")
    else:
        # If no timestamp is provided, query all logs.
        log_filter = base_filter
        print(f"Querying for all structured logs in '{SHORT_LOG_NAME}' from the beginning of time.")

    log_entries = []

    for entry in logging_client.list_entries(filter_=log_filter):
        # The Cloud Logging API can return different entry types.
        # JsonEntry has its payload in `json_payload`.
        # StructEntry has its payload in `payload`.
        # We need to handle both to be robust.
        if hasattr(entry, 'json_payload'):
            payload = entry.json_payload
        elif hasattr(entry, 'payload') and isinstance(entry.payload, dict):
            payload = entry.payload
        else:
            continue # Skip entries that are not structured or don't have a payload.
        # We need prompt and response for all metrics. For reference-based metrics
        # like ROUGE, we also look for a 'ground_truth' field.
        if all(k in payload for k in ['prompt', 'response']):
            log_entries.append({
                "prompt": payload['prompt'],
                "response": payload['response'],
                # The evaluation service expects the reference column to be named 'reference'.
                # Use `or ''` to handle cases where the 'ground_truth' key exists but its value is None (null in JSON).
                "reference": payload.get('ground_truth') or ''
            })
    
    if not log_entries:
        print("No new structured log entries found for evaluation. Exiting.")
        return None

    return pd.DataFrame(log_entries)

def run_evaluation(event=None, context=None, all_time=False):
    """
    Main function to run the evaluation.
    Accepts an 'all_time' flag to override the timestamp logic.
    """
    # FIX: Generate a unique experiment name with a timestamp
    current_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Use hyphens instead of underscores to match the required regex
    experiment_name = f"gemini-playground-{current_time_str}"
    
    # This must be called first to set the project, location, and experiment context.
    aiplatform.init(project=PROJECT_ID, location=LOCATION, experiment=experiment_name)
    
    last_run = None # Initialize to None
    if not all_time:
        last_run = get_last_run_timestamp()
    else:
        print("--- Running in --all-time mode. Evaluating all historical logs. ---")

    eval_df = get_logs_for_evaluation(last_run)
    
    if eval_df is None or eval_df.empty:
        return

    print(f"Found {len(eval_df)} new log entries to evaluate.")

    # The start_run context manager requires a unique name for the run within the experiment.
    # We can reuse the timestamp to create one.
    run_name = f"eval-run-{current_time_str}"
    print(f"Starting evaluation for experiment run: '{run_name}'")
    
    # Construct the full resource name for the judgement model.
    # The evaluation service requires the full path, not just the model ID.
    full_judgement_model_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{JUDGEMENT_MODEL_NAME}"

  # Create an AutoraterConfig object with the judgement model
    autorater_config = AutoraterConfig(
        autorater_model=full_judgement_model_name
    )
    
    eval_task = EvalTask(
        dataset=eval_df,
        metrics=[
            "fluency",
            "coherence",
            "safety",
            "rouge" # Add ROUGE to the list of metrics.
        ],
        autorater_config=autorater_config,  # Use the correct parameter as per documentation
    )

    # Let the evaluate() method create the experiment run.
    # This avoids the conflict of having a pre-existing active run.
    evaluation_result = eval_task.evaluate(
        experiment_run_name=run_name # Pass the run name to the evaluate method
    )

    # After evaluation, resume the run to log custom metrics and artifacts.
    with aiplatform.start_run(run=run_name, resume=True) as resumed_run:
        print(f"Resuming run '{run_name}' to log custom metrics and artifacts.")
        # Log the summary metrics directly to the Vertex AI Experiment run.
        summary_metrics = evaluation_result.summary_metrics
        print(f"Logging summary metrics to Vertex AI Experiment: {summary_metrics}")
        resumed_run.log_metrics(summary_metrics)

        # --- Log the detailed, per-prompt metrics table as a CSV artifact ---
        metrics_df = evaluation_result.metrics_table
        if not metrics_df.empty:
            # Define a local path for the CSV to use with log_artifact
            local_csv_path = f"per_prompt_metrics_{current_time_str}.csv"
            print(f"\n--- Saving Per-Prompt Metrics locally to: {local_csv_path} ---")
            print(metrics_df.to_string()) # Also print to console for immediate feedback
            metrics_df.to_csv(local_csv_path, index=False)

            # The eval_task.evaluate() method automatically logs the metrics_table DataFrame
            # as a CSV artifact to the experiment run. This manual step is not needed and
            # was causing the AttributeError.
            print("Per-prompt metrics table was already logged as an artifact by the evaluation service.")

        # The SDK returns summary metric keys like 'fluency/mean', 'rouge/mean', etc.
        # We filter for these keys to build the radar chart.
        labels = [key for key in summary_metrics.keys() if "/mean" in key]
        scores = [summary_metrics[key] for key in labels]

        # For a cleaner chart, remove the '/mean' suffix from the labels.
        clean_labels = [label.replace('/mean', '') for label in labels]

        if clean_labels and scores:
            print("Generating and logging radar chart artifact.")
            # --- Create and log the radar chart ---
            # The number of variables we're plotting.
            num_vars = len(clean_labels)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # The plot is a circle, so we need to "complete the loop"
            # and append the start value to the end.
            scores += scores[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.plot(angles, scores, linewidth=2, linestyle='solid', label='Model Performance')
            ax.fill(angles, scores, 'b', alpha=0.1)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(clean_labels)
            ax.set_title('Evaluation of Gemini Run', size=12, color='black', va='center')
            ax.grid(True)
            
            # Define the local and GCS paths
            image_filename = f'gemini_evaluation_radar_chart_{current_time_str}.png'
            local_image_path = os.path.join(os.getcwd(), image_filename)
            gcs_uri = f'gs://{BUCKET_NAME}/eval-artifacts/{experiment_name}/{run_name}/{image_filename}'

            # Save the chart to a local file
            plt.savefig(local_image_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            # Upload the local file to GCS
            storage_client = storage.Client(project=PROJECT_ID)
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(f'eval-artifacts/{experiment_name}/{run_name}/{image_filename}')
            blob.upload_from_filename(local_image_path)
            print(f"Uploaded radar chart to GCS: {gcs_uri}")
            
            # Log the artifact by creating the Artifact object.
            # This method directly interacts with the Metadata Store API
            # and does not rely on the `log_artifact()` helper method.
            aiplatform.Artifact.create(
                schema_title="system.Artifact",
                uri=gcs_uri,
                display_name="Radar Chart",
            )
            print("Radar chart artifact successfully associated with the experiment run.")
        else:
            print("No summary scores found to generate a radar chart. Skipping chart creation.")

    if not all_time:
        save_current_timestamp()
        print("Script finished. The last run timestamp has been updated.")
    else:
        print("Script finished. --all-time mode: last run timestamp was not updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on Gemini prompt outputs logged in Cloud Logging.")
    parser.add_argument(
        "--all-time",
        action="store_true",
        help="If set, ignores the last run timestamp and evaluates all logs from the beginning of time."
    )
    args = parser.parse_args()

    print("Running script from the command line.")
    # Pass the parsed command-line argument to the main function
    run_evaluation(all_time=args.all_time)