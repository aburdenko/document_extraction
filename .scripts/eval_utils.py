#!/usr/bin/env python3
"""
Shared utilities for running on-demand evaluations and logging to Vertex AI Experiments.
"""

import os
import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from google.cloud import storage
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from vertexai.preview.evaluation import EvalTask, AutoraterConfig

# --- Constants ---
PROJECT_ID = os.environ.get("PROJECT_ID", "your-gcp-project-id")
LOCATION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("STAGING_GCS_BUCKET", "your-bucket-name")
JUDGEMENT_MODEL_NAME = os.environ.get("JUDGEMENT_MODEL_NAME", "gemini-1.5-flash")
EXPERIMENT_NAME = "gemini-playground-evaluation"

logger = logging.getLogger(__name__)

def _generate_and_log_radar_chart(summary_metrics: dict, run_name: str, resumed_run: "aiplatform.ExperimentRun") -> Tuple[Optional[str], Optional[str]]:
    """Generates a summary radar chart, uploads it, and logs it as an artifact."""
    logger.info("    Generating and logging summary radar chart artifact.")
    labels = [key for key in summary_metrics.keys() if "/mean" in key]
    scores = [summary_metrics[key] for key in labels]
    clean_labels = [label.replace('/mean', '') for label in labels]

    if not clean_labels or not scores:
        logger.warning("    No summary scores found to generate a radar chart.")
        return None, None

    num_vars = len(clean_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, linewidth=2, linestyle='solid', label='Model Performance')
    ax.fill(angles, scores, 'b', alpha=0.1)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(clean_labels)
    ax.set_title('On-Demand Evaluation Summary', size=12, color='black', va='center')
    ax.grid(True)

    try:
        pic_io = io.BytesIO()
        plt.savefig(pic_io, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        pic_io.seek(0)
        base64_png = base64.b64encode(pic_io.getvalue()).decode('utf-8')
        html_content = f'<img src="data:image/png;base64,{base64_png}" />'

        current_time_str = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        gcs_path = f"eval-artifacts/{EXPERIMENT_NAME}/{run_name}"
        gcs_filename = f"summary-radar-chart-{current_time_str}.html"
        gcs_uri = f"gs://{BUCKET_NAME}/{gcs_path}/{gcs_filename}"
        
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(os.path.join(gcs_path, gcs_filename))
        blob.upload_from_string(html_content, content_type='text/html')
        logger.info(f"    Uploaded summary chart HTML to GCS: {gcs_uri}")

        client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
        metadata_client = aiplatform_v1.MetadataServiceClient(client_options=client_options)
        parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
        artifact_id = f"summary-radar-chart-{current_time_str}"
        artifact_to_create = aiplatform_v1.Artifact(display_name=f"summary-radar-chart-{current_time_str}", uri=gcs_uri, schema_title="system.Artifact")
        created_artifact = metadata_client.create_artifact(parent=parent_store, artifact=artifact_to_create, artifact_id=artifact_id)
        add_artifacts_request = aiplatform_v1.AddContextArtifactsAndExecutionsRequest(context=resumed_run.resource_name, artifacts=[created_artifact.name])
        metadata_client.add_context_artifacts_and_executions(request=add_artifacts_request)
        logger.info("    Successfully logged radar chart artifact to experiment run.")

        return html_content, gcs_uri

    except Exception as e:
        logger.error(f"    Failed to generate or log radar chart: {e}", exc_info=True)
        return None, None

def _log_metrics_csv_artifact(metrics_df: pd.DataFrame, run_name: str, resumed_run: "aiplatform.ExperimentRun"):
    """Logs the per-prompt metrics DataFrame as a CSV artifact."""
    if metrics_df.empty:
        logger.warning("    Metrics DataFrame is empty. Skipping CSV artifact logging.")
        return
    
    logger.info("    Logging per-prompt metrics as CSV artifact.")
    try:
        current_time_str = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        gcs_path = f"eval-artifacts/{EXPERIMENT_NAME}/{run_name}"
        gcs_filename = f"per-prompt-metrics-{current_time_str}.csv"
        gcs_uri = f"gs://{BUCKET_NAME}/{gcs_path}/{gcs_filename}"

        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(os.path.join(gcs_path, gcs_filename))
        
        blob.upload_from_string(metrics_df.to_csv(index=False), 'text/csv')
        logger.info(f"    Uploaded metrics CSV to GCS: {gcs_uri}")

        client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
        metadata_client = aiplatform_v1.MetadataServiceClient(client_options=client_options)

        parent_store = "/".join(resumed_run.resource_name.split('/')[:-2])
        artifact_id = f"per-prompt-metrics-table-{current_time_str}"

        artifact_to_create = aiplatform_v1.Artifact(
            display_name="per-prompt-metrics-table",
            uri=gcs_uri,
            schema_title="system.Artifact"
        )
        created_artifact = metadata_client.create_artifact(parent=parent_store, artifact=artifact_to_create, artifact_id=artifact_id)
        add_artifacts_request = aiplatform_v1.AddContextArtifactsAndExecutionsRequest(context=resumed_run.resource_name, artifacts=[created_artifact.name])
        metadata_client.add_context_artifacts_and_executions(request=add_artifacts_request)
        logger.info("    Successfully logged CSV artifact to experiment run.")
    except Exception as e:
        logger.error(f"    Failed to log metrics CSV artifact: {e}", exc_info=True)

def run_on_demand_evaluation(
    initial_prompt: str,
    final_answer: str,
    ground_truth: str,
    eval_metrics_list: List[str],
    filepath_stem: str,
    run_type: str = "eval-run"
) -> str:
    """
    Runs on-demand evaluation, logs to Vertex AI Experiments, and returns a markdown section.
    """
    logger.info("--- On-demand Evaluation & Experiment Logging ---")
    
    metrics_requiring_gt = ['rouge']
    requires_gt = any(m in eval_metrics_list for m in metrics_requiring_gt)
    
    if requires_gt and not ground_truth:
        logger.warning("    On-demand evaluation skipped: One or more specified metrics require a '# Ground Truth' section, which was not found.")
        return f"\n\n## Eval Output\n\n(On-demand evaluation skipped: Metrics like `{', '.join(metrics_requiring_gt)}` require a '# Ground Truth' section, which was not found.)\n"
    
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION, experiment=EXPERIMENT_NAME)
        current_time_str = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        sanitized_stem = re.sub(r'[^a-zA-Z0-9-]', '-', filepath_stem)
        run_name = f"{run_type}-{sanitized_stem}-{current_time_str}"

        logger.info(f"    Creating new experiment run: '{run_name}'")
        current_run_df = pd.DataFrame([{"prompt": initial_prompt, "response": final_answer, "reference": ground_truth or ''}])
        
        full_judgement_model_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{JUDGEMENT_MODEL_NAME}"
        autorater_config = AutoraterConfig(autorater_model=full_judgement_model_name)
        
        eval_task = EvalTask(dataset=current_run_df, metrics=eval_metrics_list, autorater_config=autorater_config)

        evaluation_result = eval_task.evaluate(experiment_run_name=run_name)
        logger.info("    Evaluation task completed and logged to new experiment run.")

        with aiplatform.start_run(run=run_name, resume=True) as resumed_run:
            summary_metrics = evaluation_result.summary_metrics
            resumed_run.log_metrics(summary_metrics)
            logger.info(f"    Logged summary metrics: {summary_metrics}")

            metrics_table_df = evaluation_result.metrics_table
            _log_metrics_csv_artifact(metrics_table_df, run_name, resumed_run)
            
            html_chart, gcs_uri = _generate_and_log_radar_chart(summary_metrics, run_name, resumed_run)

            eval_output_section = "\n\n## Eval Output\n\n"
            if html_chart:
                eval_output_section += f"{html_chart}\n"
            
            if summary_metrics:
                eval_output_section += "\n### Summary Metrics\n\n"
                eval_output_section += "| Metric | Value |\n"
                eval_output_section += "|--------|-------|\n"
                for key, value in summary_metrics.items():
                    if isinstance(value, (float, np.floating)):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    eval_output_section += f"| `{key}` | `{value_str}` |\n"
                eval_output_section += "\n"

            run_url = f"https://console.cloud.google.com/vertex-ai/experiments/locations/{LOCATION}/experiments/{EXPERIMENT_NAME}/runs/{run_name}/charts?project={PROJECT_ID}"
            eval_output_section += f"\nView full evaluation results in Vertex AI Experiments\n\n"
        
        return eval_output_section

    except Exception as e:
        logger.error(f"    Failed to run evaluation or log artifacts: {e}", exc_info=True)
        return f"\n\n## Eval Output\n\n(Failed to execute evaluation and log to Vertex AI Experiments: {e})\n"