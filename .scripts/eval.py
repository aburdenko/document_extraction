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
from vertexai.preview.evaluation import EvalTask, AutoraterConfig # This is the high-level SDK

from google.cloud import storage
from google.api_core import exceptions as google_exceptions
from google.cloud import logging
from datetime import datetime, timedelta
import os, argparse
from urllib.parse import urlparse
import re

# We need to import the low-level GAPIC client to work around SDK version issues.
from google.cloud import aiplatform_v1
# --- 1. Configuration (now using environment variables) ---
import io
import base64
# --- These variables must be set in your Cloud Function environment ---
PROJECT_ID = os.environ.get("PROJECT_ID", "your-gcp-project-id")
LOCATION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("STAGING_GCS_BUCKET", "your-bucket-name")
# The LOG_NAME env var contains the short name of the log (e.g., 'agentspace_hcls_demo_log').
# The Cloud Logging filter requires the full resource name.
SHORT_LOG_NAME = os.environ.get("LOG_NAME", "run_gemini_from_file") # Default must match run_gemini_from_file.py
EXPERIMENT_NAME = "gemini-playground-evaluation" # Use a fixed name to group all evaluations
LOG_NAME = f"projects/{PROJECT_ID}/logs/{SHORT_LOG_NAME}"
JUDGEMENT_MODEL_NAME = os.environ.get("JUDGEMENT_MODEL_NAME", "gemini-1.5-flash")

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
        # Return None but do not print here. The calling function will handle the message.
        return None 

    return pd.DataFrame(log_entries)

def cleanup_previous_artifacts(experiment_name: str):
    """
    Finds all but the most recent run in an experiment and deletes their
    custom artifacts (radar chart and metrics table) from both the
    Vertex AI Metadata store and GCS to keep the experiment clean.
    """
    try:
        print("--- Starting cleanup of previous evaluation artifacts (Metadata & GCS) ---")
        # Use the static list method on ExperimentRun, which is more stable across SDK versions.
        runs = aiplatform.ExperimentRun.list(experiment=experiment_name)

        if len(runs) <= 1:
            print("No previous runs found to clean up.")
            return

        # Sort runs by creation time, descending, to find the latest.
        runs.sort(key=lambda r: r.create_time, reverse=True)
        
        latest_run = runs[0]
        previous_runs = runs[1:]
        
        print(f"Keeping artifacts for latest run: '{latest_run.name}' (created at {latest_run.create_time}).")
        print(f"Found {len(previous_runs)} previous runs to clean up artifacts from.")

        storage_client = storage.Client(project=PROJECT_ID)

        for run in previous_runs:
            print(f"  Processing previous run: '{run.name}'")
            artifacts_to_delete = []
            for artifact in run.get_artifacts():
                # Find artifacts by their display name
                if (artifact.display_name.startswith("radar-chart-") or
                    artifact.display_name == "per-prompt-metrics-table" or
                    artifact.display_name == "per-prompt-radar-chart"):
                    artifacts_to_delete.append(artifact)
            if not artifacts_to_delete:
                print(f"    No matching artifacts found in run '{run.name}'.")
                continue

            for artifact in artifacts_to_delete:
                gcs_uri = artifact.uri
                print(f"    Deleting artifact metadata: '{artifact.display_name}' ({artifact.resource_name})")
                try:
                    artifact.delete()
                except Exception as e:
                    print(f"    Warning: Failed to delete artifact metadata '{artifact.display_name}': {e}")
                
                # Also delete the underlying GCS file
                if gcs_uri and gcs_uri.startswith("gs://"):
                    try:
                        parsed_uri = urlparse(gcs_uri)
                        bucket = storage_client.bucket(parsed_uri.netloc)
                        blob = bucket.blob(parsed_uri.path.lstrip('/'))
                        if blob.exists():
                            print(f"    Deleting associated GCS file: {gcs_uri}")
                            blob.delete()
                    except Exception as e:
                        print(f"    Warning: Failed to delete GCS file {gcs_uri}: {e}")
        print("--- Cleanup complete ---")
    except google_exceptions.NotFound:
        print(f"Experiment '{experiment_name}' not found. This is expected on the very first run. Skipping cleanup.")
    except Exception as e:
        print(f"Warning: An error occurred during artifact cleanup: {e}. This may be expected on the first run.")

def _generate_per_prompt_radar_chart(metrics_df: pd.DataFrame, metrics: list, run_name: str, experiment_name: str, current_time_str: str, resumed_run: aiplatform.ExperimentRun):
    """
    Generates and logs a radar chart showing metrics for each prompt in a run.
    """
    print("Generating and logging per-prompt radar chart artifact.")

    # --- Add diagnostic logging to inspect the DataFrame ---
    print("--- Per-Prompt Metrics Table Diagnostics ---")
    print("Columns available in metrics_table:", metrics_df.columns.tolist())
    print("Data types of columns:\n", metrics_df.dtypes)
    print("First 2 rows of metrics_table:")
    # Use to_string() to prevent truncation of wide columns
    print(metrics_df.head(2).to_string())
    print("------------------------------------------")

    # The 'metrics' list comes from the cleaned summary metrics keys (e.g., ['fluency', 'rouge']).
    # The evaluation service may return autorater scores (fluency, etc.) as a dictionary
    # in a column with the metric's name. We need to dynamically extract the 'rating' value.
    
    # Create a copy to avoid SettingWithCopyWarning
    processed_df = metrics_df.copy()
    metric_cols_to_plot = []
    final_labels_for_chart = []

    for metric in metrics: # e.g., 'fluency', 'coherence', 'safety', 'rouge'
        # The detailed metrics table from the SDK uses a 'metric/score' naming convention.
        score_col_name = f"{metric}/score"
        if score_col_name in processed_df.columns:
            # This column exists, now we need to process it for the 0-1 scale of the radar chart.
            
            # ROUGE is already 0-1, so we can use it directly.
            if metric == 'rouge':
                # No normalization needed.
                metric_cols_to_plot.append(score_col_name)
                final_labels_for_chart.append(metric) # Use the clean name for the label
            else:
                # This is an autorater metric. Normalize it to 0-1.
                normalized_col_name = f"{metric}_normalized"
                print(f"Normalizing autorater metric '{metric}' from column '{score_col_name}' into '{normalized_col_name}'.")
                
                if metric in ['fluency', 'coherence']:
                    # Normalize from [1, 5] to [0, 1] where 5 is best.
                    # Formula: (score - 1) / (5 - 1)
                    processed_df[normalized_col_name] = (processed_df[score_col_name] - 1.0) / 4.0
                elif metric == 'safety':
                    # Safety score is 1-4, where 1 is best (no concerns).
                    # We need to invert and normalize it so that 1 (best) -> 1.0 and 4 (worst) -> 0.0.
                    # Formula: (4 - score) / 3
                    processed_df[normalized_col_name] = (4.0 - processed_df[score_col_name]) / 3.0
                else:
                    # Default to 1-5 scale for any other unknown autorater metric
                    print(f"Warning: Unhandled autorater metric '{metric}'. Assuming a 1-5 scale for normalization.")
                    processed_df[normalized_col_name] = (processed_df[score_col_name] - 1.0) / 4.0
                
                # Clip values to be strictly between 0 and 1 in case of any oddities.
                processed_df[normalized_col_name] = processed_df[normalized_col_name].clip(0, 1)
                
                metric_cols_to_plot.append(normalized_col_name)
                final_labels_for_chart.append(metric) # Use the clean name for the label
        else:
            print(f"Warning: Expected metric column '{score_col_name}' not found.")

    if not metric_cols_to_plot:
        print("No valid metric columns could be processed for the per-prompt radar chart. Skipping.")
        return

    print(f"Final list of metric columns for plotting: {metric_cols_to_plot}")
    labels = final_labels_for_chart
    num_vars = len(labels)

    # Compute angle for each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop.

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Use a colormap to differentiate prompts.
    colors = plt.cm.get_cmap('viridis', len(processed_df))

    for i, row in processed_df.iterrows():
        # Ensure we only try to plot if all metric columns are present and not NaN for the row
        if not all(m in row and pd.notna(row[m]) for m in metric_cols_to_plot):
            continue

        scores = row[metric_cols_to_plot].values.tolist()
        scores += scores[:1]  # Complete the loop.

        # Create a short label for the legend.
        prompt_label = f"Prompt {i+1}: {row['prompt'][:40]}..."

        ax.plot(angles, scores, color=colors(i), linewidth=1.5, linestyle='solid', label=prompt_label)

    # --- Format the plot ---
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(30)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_title('Per-Prompt Evaluation Metrics', size=16, color='black', y=1.1)
    ax.grid(True)

    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))

    latest_png_filename = "per-prompt-radar-chart-latest.png"
    try:
        plt.savefig(latest_png_filename, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved latest per-prompt radar chart for preview: {latest_png_filename}")
    except Exception as e:
        print(f"Warning: Could not save latest per-prompt radar chart PNG: {e}")

    pic_io = io.BytesIO()
    plt.savefig(pic_io, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    pic_io.seek(0)
    base64_png = base64.b64encode(pic_io.read()).decode('utf-8')
    html_content = f'<html><body><img src="data:image/png;base64,{base64_png}" /></body></html>'

    try:
        gcs_path = f"eval-artifacts/{experiment_name}/{run_name}"
        gcs_html_filename = f"per-prompt-radar-chart-{current_time_str}.html"
        gcs_uri = f"gs://{BUCKET_NAME}/{gcs_path}/{gcs_html_filename}"
        blob = storage.Client(project=PROJECT_ID).bucket(BUCKET_NAME).blob(os.path.join(gcs_path, gcs_html_filename))
        blob.upload_from_string(html_content, content_type='text/html') # Upload the HTML content to GCS
        print(f"Uploaded per-prompt radar chart HTML to staging GCS: {gcs_uri}")

        # --- Use the GAPIC client to log the artifact, which is the correct pattern ---
        client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
        metadata_client = aiplatform_v1.MetadataServiceClient(client_options=client_options)
        parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
        artifact_id = f"per-prompt-radar-chart-{current_time_str}"
        artifact_to_create = aiplatform_v1.Artifact(
            display_name="per-prompt-radar-chart",
            uri=gcs_uri,
            schema_title="system.html"
        )
        created_artifact = metadata_client.create_artifact(parent=parent_store, artifact=artifact_to_create, artifact_id=artifact_id)
        add_artifacts_request = aiplatform_v1.AddContextArtifactsAndExecutionsRequest(context=resumed_run.resource_name, artifacts=[created_artifact.name])
        metadata_client.add_context_artifacts_and_executions(request=add_artifacts_request)
        print("Per-prompt radar chart artifact successfully associated with the run via GAPIC client.")
    except Exception as e:
        print(f"An error occurred while logging the per-prompt radar chart artifact: {e}")

def run_evaluation(event=None, context=None, all_time=False):
    """
    Main function to run the evaluation.
    Accepts an 'all_time' flag to override the timestamp logic.
    """
    current_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Use a fixed experiment name to group all evaluations together.
    experiment_name = EXPERIMENT_NAME
    
    # This must be called first to set the project, location, and experiment context.
    aiplatform.init(project=PROJECT_ID, location=LOCATION, experiment=experiment_name)

    # Before running a new evaluation, clean up the artifacts from all but the most recent previous run.
    cleanup_previous_artifacts(experiment_name)
    
    last_run = None # Initialize to None
    if not all_time:
        last_run = get_last_run_timestamp()
    else:
        print("--- Running in --all-time mode. Evaluating all historical logs. ---")

    eval_df = get_logs_for_evaluation(last_run)
    
    if eval_df is None or eval_df.empty:
        print("No new structured log entries found for evaluation.")
        # If not running in --all-time mode, suggest it as a possible solution.
        if not all_time:
            print("Tip: To evaluate all historical logs, try running with the --all-time flag.")
        print("Exiting.")
        return # Exit the function

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
        autorater_config=autorater_config,
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

        # The eval_task.evaluate() method automatically logs the metrics_table DataFrame
        # as a CSV artifact to the experiment run. This comment was incorrect; we need to log it explicitly.
        # print("Per-prompt metrics table was already logged as an artifact by the evaluation service.")

        # --- Explicitly log the per-prompt metrics table as a CSV artifact ---
        # The evaluate() method returns the metrics table, but it's not always automatically
        # logged as an artifact. We do it here to ensure it appears in the UI.
        try:
            print("Explicitly logging per-prompt metrics table as a CSV artifact...")
            metrics_df = evaluation_result.metrics_table
            if not metrics_df.empty:
                gcs_path = f"eval-artifacts/{experiment_name}/{run_name}"
                # Use a timestamp for the GCS artifact to ensure uniqueness in the experiment run
                gcs_csv_filename = f"per-prompt-metrics-{current_time_str}.csv"
                # Use a static name for the local file for easy preview, which will be overwritten
                local_csv_filename = "per-prompt-metrics-latest.csv"
                local_csv_path = os.path.join(os.getcwd(), local_csv_filename)

                # Write to a local file first.
                metrics_df.to_csv(local_csv_path, index=False)
                print(f"Saved latest per-prompt metrics for preview: {local_csv_filename}")

                # Upload the local file to GCS with a unique name.
                gcs_uri = f"gs://{BUCKET_NAME}/{gcs_path}/{gcs_csv_filename}"
                storage_client = storage.Client(project=PROJECT_ID)
                bucket = storage_client.bucket(BUCKET_NAME)
                blob = bucket.blob(os.path.join(gcs_path, gcs_csv_filename))
                blob.upload_from_filename(local_csv_path, content_type='text/csv')
                print(f"Uploaded per-prompt metrics CSV to staging GCS: {gcs_uri}")

                # Log the artifact to the experiment run.
                client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
                metadata_client = aiplatform_v1.MetadataServiceClient(client_options=client_options)
                parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
                artifact_id = f"per-prompt-metrics-{current_time_str}"
                artifact_to_create = aiplatform_v1.Artifact(display_name="per-prompt-metrics-table", uri=gcs_uri, schema_title="system.Artifact")
                created_artifact = metadata_client.create_artifact(parent=parent_store, artifact=artifact_to_create, artifact_id=artifact_id)
                add_artifacts_request = aiplatform_v1.AddContextArtifactsAndExecutionsRequest(context=resumed_run.resource_name, artifacts=[created_artifact.name])
                metadata_client.add_context_artifacts_and_executions(request=add_artifacts_request)
                print("Per-prompt metrics artifact successfully associated with the run.")

        except Exception as e:
            print(f"An error occurred while logging the per-prompt metrics artifact: {e}")

        # --- Remove the auto-generated TensorBoard run artifact ---
        # The evaluate() method automatically creates a TensorBoard run artifact for visualization,
        # which can add clutter if not needed. We will find and delete this specific artifact.
        try:
            for artifact in resumed_run.get_artifacts():
                # The most reliable way to identify the TensorBoard run artifact is by its schema title,
                # which is set by the evaluation service. Using the display name can be brittle.
                # The correct schema for this auto-generated artifact is 'google.cloud.aiplatform.v1.TensorboardRun'.
                if artifact.schema_title == 'google.cloud.aiplatform.v1.TensorboardRun':
                    print(f"Found and deleting auto-generated TensorBoard artifact: {artifact.display_name}")
                    artifact.delete()
                    print("TensorBoard artifact deleted.")
                    # We assume there's only one per run, so we can break the loop.
                    break
        except Exception as e:
            print(f"Could not delete TensorBoard artifact: {e}")

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

            # --- Save the plot to a local PNG file for easy preview ---
            latest_png_filename = "radar-chart-latest.png"
            try:
                # Save to a file before closing the figure
                plt.savefig(latest_png_filename, format='png', bbox_inches='tight', dpi=150)
                print(f"Saved latest radar chart for preview: {latest_png_filename}")
            except Exception as e:
                print(f"Warning: Could not save latest radar chart PNG: {e}")

            # 1. Save the plot to an in-memory buffer and create HTML content.
            pic_io = io.BytesIO()
            plt.savefig(pic_io, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig) # Close the figure to free memory
            pic_io.seek(0)
            base64_png = base64.b64encode(pic_io.read()).decode('utf-8')
            html_content = f'<html><body><img src="data:image/png;base64,{base64_png}" /></body></html>'

            # --- Create and log the radar chart as an inline HTML artifact ---
            try:
                gcs_path = f"eval-artifacts/{experiment_name}/{run_name}"
                gcs_html_filename = f"radar-chart-{current_time_str}.html"
                local_html_filename = "radar-chart-latest.html"
                local_html_path = os.path.join(os.getcwd(), local_html_filename)

                with open(local_html_path, "w") as f:
                    f.write(html_content)
                print(f"Saved latest radar chart HTML for preview: {local_html_filename}")

                gcs_uri = f"gs://{BUCKET_NAME}/{gcs_path}/{gcs_html_filename}"
                storage_client = storage.Client(project=PROJECT_ID)
                bucket = storage_client.bucket(BUCKET_NAME)
                blob = bucket.blob(os.path.join(gcs_path, gcs_html_filename))
                blob.upload_from_filename(local_html_path, content_type='text/html')
                print(f"Uploaded radar chart HTML to staging GCS: {gcs_uri}")

                client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
                metadata_client = aiplatform_v1.MetadataServiceClient(client_options=client_options)
                parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
                artifact_id = f"radar-chart-{current_time_str}"
                artifact_to_create = aiplatform_v1.Artifact(display_name=f"radar-chart-{current_time_str}", uri=gcs_uri, schema_title="system.html")
                created_artifact = metadata_client.create_artifact(parent=parent_store, artifact=artifact_to_create, artifact_id=artifact_id)
                add_artifacts_request = aiplatform_v1.AddContextArtifactsAndExecutionsRequest(context=resumed_run.resource_name, artifacts=[created_artifact.name])
                metadata_client.add_context_artifacts_and_executions(request=add_artifacts_request)
                print("Radar chart artifact successfully associated with the run via GAPIC client.")

            except Exception as e:
                print(f"An error occurred while logging the artifact with the GAPIC client: {e}")
        else:
            print("No summary scores found to generate a radar chart. Skipping chart creation.")

        # --- Generate and log the new per-prompt radar chart ---
        metrics_df = evaluation_result.metrics_table
        if not metrics_df.empty and clean_labels:
            # The metric names for the chart axes are the same as the summary chart's clean_labels
            _generate_per_prompt_radar_chart(
                metrics_df=metrics_df,
                metrics=clean_labels, # Pass the derived metric names
                run_name=run_name,
                experiment_name=experiment_name,
                current_time_str=current_time_str,
                resumed_run=resumed_run
            )


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