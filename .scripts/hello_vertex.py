#!/usr/bin/env python3
"""
A minimal "hello world" script for testing Vertex AI Custom Job submission.
"""
import os
import time

print("--- Hello from Vertex AI Custom Job! ---")
print("This script will print some environment variables and then exit.")

# Print some of the environment variables passed from the run_as_pipeline.py script
project_id = os.getenv("PROJECT_ID")
region = os.getenv("REGION")

print(f"  PROJECT_ID: {project_id}")
print(f"  REGION: {region}")

print("\nSleeping for 15 seconds to make the job visible in the console...")
time.sleep(15)

print("\nScript finished successfully.")