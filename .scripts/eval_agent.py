#!/usr/bin/env python3
# Run with:
# .scripts/eval.py
# .scripts/eval.py --all-time

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.cloud import aiplatform
import vertexai
from vertexai.preview.evaluation import EvalTask, AutoraterConfig, CustomMetric # This is the high-level SDK

from google.cloud import storage
from google.api_core import exceptions as google_exceptions
from google.cloud import logging
from datetime import datetime, timedelta
import os, argparse
from urllib.parse import urlparse
import re

# We need to import the low-level GAPIC client to work around SDK version issues.
from google.cloud import aiplatform_v1
import json
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

def get_logs_for_evaluation(last_run_timestamp: str | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Queries Cloud Logging for new logs since the last run.
    Separates logs into agentic (session_id-based) and simple (request_id-based).
    Returns two DataFrames: (agent_df, simple_df).
    """
    logging_client = logging.Client(project=PROJECT_ID)

    # Filter for agent logs (session_id) OR simple logs (request_id)
    base_filter = (
        f'logName="{LOG_NAME}" AND '
        f'(jsonPayload.session_id:* OR jsonPayload.request_id:*)'
    )

    if last_run_timestamp:
        log_filter = f'{base_filter} AND timestamp >= "{last_run_timestamp}"'
        print(f"Querying for agent or simple logs in '{SHORT_LOG_NAME}' since: {last_run_timestamp}")
    else:
        log_filter = base_filter
        print(f"Querying for all agent or simple logs in '{SHORT_LOG_NAME}'.")

    # --- Separate logs into two buckets ---
    agent_sessions = {}
    simple_logs = []

    for entry in logging_client.list_entries(filter_=log_filter):
        payload = entry.json_payload if hasattr(entry, 'json_payload') else {}
        
        # Prioritize agentic logs if both keys somehow exist
        if "session_id" in payload:
            session_id = payload.get("session_id")
            if session_id not in agent_sessions:
                agent_sessions[session_id] = {
                    "logs": [], "instruction": payload.get("instruction"),
                    "initial_prompt": payload.get("initial_prompt"),
                    "ground_truth": payload.get("ground_truth"), "final_answer": ""
                }
            agent_sessions[session_id]["logs"].append(payload)
            if payload.get("log_type") == "final_answer":
                agent_sessions[session_id]["final_answer"] = payload.get("final_answer")
        
        elif "request_id" in payload:
            # This is a simple, non-agentic log
            if all(k in payload for k in ['prompt', 'response']):
                simple_logs.append({
                    "prompt": payload['prompt'], "response": payload['response'],
                    "reference": payload.get('ground_truth') or ''
                })

    # --- Process Agentic Logs ---
    agent_df = None
    if agent_sessions:
        agent_eval_data = []
        for session_id, data in agent_sessions.items():
            sorted_logs = sorted(data["logs"], key=lambda x: x.get("step", 0))
            full_trace = "\n".join([json.dumps(log) for log in sorted_logs])
            agent_eval_data.append({
                "instruction": data["instruction"], "initial_prompt": data["initial_prompt"],
                "full_trace": full_trace, "final_answer": data["final_answer"],
                "reference": data["ground_truth"],
            })
        agent_df = pd.DataFrame(agent_eval_data)

    # --- Process Simple Logs ---
    simple_df = None
    if simple_logs:
        simple_df = pd.DataFrame(simple_logs)

    return agent_df, simple_df

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
        runs.sort(key=lambda r: r.start_time, reverse=True)
        
        latest_run = runs[0]
        previous_runs = runs[1:]
        
        print(f"Keeping artifacts for latest run: '{latest_run.name}' (started at {latest_run.start_time}).")
        print(f"Found {len(previous_runs)} previous runs to clean up artifacts from.")

        storage_client = storage.Client(project=PROJECT_ID)

        for run in previous_runs:
            print(f"  Processing previous run: '{run.name}'")
            artifacts_to_delete = []
            for artifact in run.get_artifacts():
                # Find artifacts by their display name
                if (artifact.display_name.startswith("radar-chart-") or
                    artifact.display_name.startswith("per-prompt-metrics-table") or
                    artifact.display_name.startswith("per-prompt-radar-chart")): # Also cleans up new artifact
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

def _generate_per_prompt_radar_chart(metrics_df: pd.DataFrame, metrics: list, run_name: str, experiment_name: str, current_time_str: str, resumed_run: aiplatform.ExperimentRun, run_name_suffix: str):
    """
    Generates and logs a radar chart showing metrics for each prompt in a run.
    (Adapted from the original eval.py script)
    """
    print("Generating and logging per-prompt radar chart artifact.")

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
                
                if metric in ['fluency', 'coherence', 'tool_call_validity', 'tool_name_match', 'tool_parameter_match', 'agent_task_quality']:
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

    labels = final_labels_for_chart
    num_vars = len(labels)

    # Compute angle for each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop.

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    colors = plt.cm.get_cmap('viridis', len(processed_df))

    for i, row in processed_df.iterrows():
        if not all(m in row and pd.notna(row[m]) for m in metric_cols_to_plot):
            continue
        scores = row[metric_cols_to_plot].values.tolist()
        scores += scores[:1]
        prompt_label = f"Prompt {i+1}: {row.get('prompt', row.get('initial_prompt', ''))[:40]}..."
        ax.plot(angles, scores, color=colors(i), linewidth=1.5, linestyle='solid', label=prompt_label)

    ax.set_ylim(0, 1)
    ax.set_rlabel_position(30)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_title(f'Per-Prompt Evaluation Metrics ({run_name_suffix.strip("-")})', size=16, color='black', y=1.1)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))

    latest_png_filename = f"per-prompt-radar-chart-latest{run_name_suffix}.png"
    plt.savefig(latest_png_filename, format='png', bbox_inches='tight', dpi=150)
    print(f"Saved latest per-prompt radar chart for preview: {latest_png_filename}")

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
        blob.upload_from_string(html_content, content_type='text/html')
        parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
        artifact_id = f"per-prompt-radar-chart-{current_time_str}{run_name_suffix}"
        artifact_to_create = aiplatform_v1.Artifact(display_name=f"per-prompt-radar-chart{run_name_suffix}", uri=gcs_uri, schema_title="system.html")
        metadata_client = aiplatform_v1.MetadataServiceClient(client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})
        created_artifact = metadata_client.create_artifact(parent=parent_store, artifact=artifact_to_create, artifact_id=artifact_id)
        add_artifacts_request = aiplatform_v1.AddContextArtifactsAndExecutionsRequest(context=resumed_run.resource_name, artifacts=[created_artifact.name])
        metadata_client.add_context_artifacts_and_executions(request=add_artifacts_request)
        print("Per-prompt radar chart artifact successfully associated with the run.")
    except Exception as e:
        print(f"An error occurred while logging the per-prompt radar chart artifact: {e}")

def _execute_evaluation_run(
    eval_df: pd.DataFrame,
    metrics: list,
    run_name_suffix: str,
    experiment_name: str,
    current_time_str: str
):
    """
    Executes a single evaluation run for a given DataFrame and metrics.
    Logs all results and artifacts to a new run in the experiment.
    """
    run_name = f"eval-run-{current_time_str}{run_name_suffix}"
    print(f"--- Starting evaluation for experiment run: '{run_name}' ---")
    
    full_judgement_model_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{JUDGEMENT_MODEL_NAME}"
    
    autorater_config = AutoraterConfig(autorater_model=full_judgement_model_name)

    eval_task = EvalTask(
        dataset=eval_df,
        metrics=metrics,
        autorater_config=autorater_config,
    )

    evaluation_result = eval_task.evaluate(
        experiment_run_name=run_name
    )

    with aiplatform.start_run(run=run_name, resume=True) as resumed_run:
        print(f"Resuming run '{run_name}' to log custom metrics and artifacts.")
        summary_metrics = evaluation_result.summary_metrics
        print(f"Logging summary metrics to Vertex AI Experiment: {summary_metrics}")
        resumed_run.log_metrics(summary_metrics)

        try:
            print("Explicitly logging per-prompt metrics table as a CSV artifact...")
            metrics_df = evaluation_result.metrics_table
            if not metrics_df.empty:
                gcs_path = f"eval-artifacts/{experiment_name}/{run_name}"
                gcs_csv_filename = f"per-prompt-metrics-{current_time_str}.csv"
                local_csv_filename = f"per-prompt-metrics-latest{run_name_suffix}.csv"
                local_csv_path = os.path.join(os.getcwd(), local_csv_filename)

                metrics_df.to_csv(local_csv_path, index=False)
                print(f"Saved latest per-prompt metrics for preview: {local_csv_filename}")

                gcs_uri = f"gs://{BUCKET_NAME}/{gcs_path}/{gcs_csv_filename}"
                storage_client = storage.Client(project=PROJECT_ID)
                bucket = storage_client.bucket(BUCKET_NAME)
                blob = bucket.blob(os.path.join(gcs_path, gcs_csv_filename))
                blob.upload_from_filename(local_csv_path, content_type='text/csv')
                print(f"Uploaded per-prompt metrics CSV to staging GCS: {gcs_uri}")

                client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
                metadata_client = aiplatform_v1.MetadataServiceClient(client_options=client_options)
                parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
                artifact_id = f"per-prompt-metrics-{current_time_str}{run_name_suffix}"
                artifact_to_create = aiplatform_v1.Artifact(display_name=f"per-prompt-metrics-table{run_name_suffix}", uri=gcs_uri, schema_title="system.Artifact")
                created_artifact = metadata_client.create_artifact(parent=parent_store, artifact=artifact_to_create, artifact_id=artifact_id)
                add_artifacts_request = aiplatform_v1.AddContextArtifactsAndExecutionsRequest(context=resumed_run.resource_name, artifacts=[created_artifact.name])
                metadata_client.add_context_artifacts_and_executions(request=add_artifacts_request)
                print("Per-prompt metrics artifact successfully associated with the run.")

        except Exception as e:
            print(f"An error occurred while logging the per-prompt metrics artifact: {e}")

        try:
            for artifact in resumed_run.get_artifacts():
                if artifact.schema_title == 'google.cloud.aiplatform.v1.TensorboardRun':
                    print(f"Found and deleting auto-generated TensorBoard artifact: {artifact.display_name}")
                    artifact.delete()
                    print("TensorBoard artifact deleted.")
                    # We assume there's only one per run, so we can break the loop.
                    break
        except Exception as e:
            print(f"Could not delete TensorBoard artifact: {e}")

        labels = [key for key in summary_metrics.keys() if "/mean" in key]
        scores = [summary_metrics[key] for key in labels]

        clean_labels = [label.replace('/mean', '') for label in labels]

        if clean_labels and scores:
            print("Generating and logging radar chart artifact.")
            num_vars = len(clean_labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            scores += scores[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.plot(angles, scores, linewidth=2, linestyle='solid', label=f'Performance ({run_name_suffix.strip("-")})')
            ax.fill(angles, scores, 'b', alpha=0.1)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(clean_labels)
            ax.set_title(f'Evaluation of Gemini Run ({run_name_suffix.strip("-")})', size=12, color='black', va='center')
            ax.grid(True)
            
            latest_png_filename = f"radar-chart-latest{run_name_suffix}.png"
            try:
                plt.savefig(latest_png_filename, format='png', bbox_inches='tight', dpi=150)
                print(f"Saved latest radar chart for preview: {latest_png_filename}")
            except Exception as e:
                print(f"Warning: Could not save latest radar chart PNG: {e}")

            pic_io = io.BytesIO()
            plt.savefig(pic_io, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            pic_io.seek(0)
            base64_png = base64.b64encode(pic_io.read()).decode('utf-8')
            html_content = f'<html><body><img src="data:image/png;base64,{base64_png}" /></body></html>'

            try:
                gcs_path = f"eval-artifacts/{experiment_name}/{run_name}"
                gcs_html_filename = f"radar-chart-{current_time_str}.html"
                local_html_filename = f"radar-chart-latest{run_name_suffix}.html"
                local_html_path = os.path.join(os.getcwd(), local_html_filename)

                with open(local_html_path, "w") as f:
                    f.write(html_content)
                print(f"Saved latest radar chart HTML for preview: {local_html_filename}")

                gcs_uri = f"gs://{BUCKET_NAME}/{gcs_path}/{gcs_html_filename}"
                blob = storage.Client(project=PROJECT_ID).bucket(BUCKET_NAME).blob(os.path.join(gcs_path, gcs_html_filename))
                blob.upload_from_filename(local_html_path, content_type='text/html')
                print(f"Uploaded radar chart HTML to staging GCS: {gcs_uri}")

                client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
                metadata_client = aiplatform_v1.MetadataServiceClient(client_options=client_options)
                parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
                artifact_id = f"radar-chart-{current_time_str}{run_name_suffix}"
                artifact_to_create = aiplatform_v1.Artifact(display_name=f"radar-chart-{current_time_str}{run_name_suffix}", uri=gcs_uri, schema_title="system.html")
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
            _generate_per_prompt_radar_chart(
                metrics_df=metrics_df,
                metrics=clean_labels,
                run_name=run_name,
                experiment_name=experiment_name,
                current_time_str=current_time_str,
                resumed_run=resumed_run,
                run_name_suffix=run_name_suffix
            )
    print(f"--- Finished evaluation for run: '{run_name}' ---")

def run_evaluation(event=None, context=None, all_time=False):
    """
    Main function to run the evaluation.
    Detects agentic and simple logs and runs separate evaluations for each.
    """
    current_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_name = EXPERIMENT_NAME
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION, experiment=experiment_name)

    cleanup_previous_artifacts(experiment_name)
    
    last_run = None
    if not all_time:
        last_run = get_last_run_timestamp()
    else:
        print("--- Running in --all-time mode. Evaluating all historical logs. ---")

    agent_df, simple_df = get_logs_for_evaluation(last_run)
    
    found_logs = False
    if agent_df is not None and not agent_df.empty:
        found_logs = True
        print(f"Found {len(agent_df)} new agentic log entries to evaluate.")
        
        task_quality_metric = CustomMetric(
            name="agent_task_quality",
            prompt=(
                "You are an expert evaluator. Given the agent's instruction, the ground truth, the final answer, and the full execution trace, "
                "please evaluate the agent's overall performance on a scale of 1-5. Consider if the agent achieved the goal, "
                "if it used tools correctly, and if its final answer is accurate and complete.\n\n"
                "Instruction: {instruction}\n"
                "Ground Truth: {reference}\n"
                "Final Answer: {final_answer}\n"
                "Execution Trace:\n{full_trace}"
            ),
            result_parser="rating_reason"
        )
        agentic_metrics = [
            task_quality_metric, "tool_call_validity", "tool_name_match",
            "tool_parameter_match", "rouge"
        ]
        
        _execute_evaluation_run(
            eval_df=agent_df,
            metrics=agentic_metrics,
            run_name_suffix="-agentic",
            experiment_name=experiment_name,
            current_time_str=current_time_str
        )

    if simple_df is not None and not simple_df.empty:
        found_logs = True
        print(f"Found {len(simple_df)} new simple log entries to evaluate.")
        
        simple_metrics = ["fluency", "coherence", "safety", "rouge"]
        
        _execute_evaluation_run(
            eval_df=simple_df,
            metrics=simple_metrics,
            run_name_suffix="-simple",
            experiment_name=experiment_name,
            current_time_str=current_time_str
        )

    if not found_logs:
        print("No new structured log entries found for evaluation.")
        if not all_time:
            print("Tip: To evaluate all historical logs, try running with the --all-time flag.")
        print("Exiting.")
        return

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