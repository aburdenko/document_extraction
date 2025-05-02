#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold, SafetySettingDict

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure API Key (replace with your actual key or environment variable handling)
# It's recommended to use environment variables for API keys
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# Default safety settings - adjust as needed
DEFAULT_SAFETY_SETTINGS: List[SafetySettingDict] = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
]

# --- Helper Functions ---

def read_file(filepath: str, base_dir: str) -> str:
    """Reads content from a file, resolving relative paths."""
    try:
        # Resolve path relative to the prompt file's directory
        absolute_filepath = os.path.join(base_dir, filepath)
        if not os.path.exists(absolute_filepath):
             # Fallback: try relative to current working directory if not found relative to prompt
             absolute_filepath = filepath
             if not os.path.exists(absolute_filepath):
                 raise FileNotFoundError(f"File not found at {os.path.join(base_dir, filepath)} or {filepath}")

        with open(absolute_filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as e:
        logging.error(f"Error: Input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        sys.exit(1)

def parse_prompt_markdown(filepath: str) -> Tuple[Dict[str, Any], Optional[str], Optional[str], str, Optional[str]]:
    """
    Parses the prompt markdown file to extract metadata, system instructions,
    schema, functions, and the main prompt body.
    """
    metadata = {}
    system_instructions = None
    schema_str = None
    functions_str = None # Placeholder for potential future function calling
    prompt_body_lines = []
    current_section = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith('# '):
                    section_title = stripped_line[2:].strip().lower()
                    if section_title == 'metadata':
                        current_section = 'metadata'
                    elif section_title == 'system instructions':
                        current_section = 'system_instructions'
                        system_instructions = ""
                    elif section_title == 'controlled output schema':
                        current_section = 'schema'
                        schema_str = ""
                    elif section_title == 'functions': # Placeholder
                        current_section = 'functions'
                        functions_str = ""
                    elif section_title == 'prompt':
                        current_section = 'prompt'
                    else:
                        # If it's a different H1, assume it's part of the prompt body
                        current_section = 'prompt'
                        prompt_body_lines.append(line)
                elif current_section == 'metadata':
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                elif current_section == 'system_instructions':
                    system_instructions += line
                elif current_section == 'schema':
                    schema_str += line
                elif current_section == 'functions':
                    functions_str += line
                elif current_section == 'prompt':
                    prompt_body_lines.append(line)
                # Ignore lines before the first section marker

        prompt_body = "".join(prompt_body_lines).strip()
        if system_instructions:
            system_instructions = system_instructions.strip()
        if schema_str:
            schema_str = schema_str.strip()
        if functions_str:
            functions_str = functions_str.strip() # Placeholder

        return metadata, system_instructions, schema_str, prompt_body, functions_str

    except FileNotFoundError:
        logging.error(f"Error: Prompt file not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error parsing prompt file {filepath}: {e}")
        sys.exit(1)

def parse_schema(schema_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parses the JSON schema string into a dictionary."""
    if not schema_str:
        return None
    try:
        # Remove potential markdown code fences
        schema_str = re.sub(r'^```json\s*', '', schema_str, flags=re.MULTILINE)
        schema_str = re.sub(r'\s*```$', '', schema_str, flags=re.MULTILINE)
        return json.loads(schema_str)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON schema: {e}\nSchema content:\n{schema_str}")
        return None # Continue without schema if parsing fails, but log error
    except Exception as e:
        logging.error(f"An unexpected error occurred during schema parsing: {e}")
        return None

def generate_content(
    model_name: str,
    prompt: str,
    temperature: float,
    system_instruction: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    safety_settings: Optional[List[SafetySettingDict]] = None,
    # functions: Optional[List[Any]] = None # Placeholder for tools/functions
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Generates content using the Gemini API."""
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            safety_settings=safety_settings or DEFAULT_SAFETY_SETTINGS
        )

        generation_config = GenerationConfig(
            temperature=temperature,
            # Add response_mime_type if schema is provided for JSON mode
            response_mime_type="application/json" if schema else None,
            response_schema=schema if schema else None,
            # candidate_count=1 # Usually default, but can be explicit
            # max_output_tokens=... # Optional: set limits if needed
            # top_p=... # Optional: nucleus sampling
            # top_k=... # Optional: top-k sampling
        )

        logging.info(f"Sending request to model: {model_name} with temp={temperature}")
        if schema:
            logging.info("JSON output mode enabled with schema.")
        if system_instruction:
            logging.info("System instructions provided.")

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            # tools=functions # Placeholder for function calling
        )

        # Extract usage metadata if available
        prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else None
        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None

        # Handle potential blocked responses
        if not response.candidates:
             block_reason = prompt_feedback.block_reason.name if prompt_feedback and prompt_feedback.block_reason else "Unknown"
             logging.warning(f"Response was blocked. Reason: {block_reason}")
             # You might want to inspect prompt_feedback.safety_ratings here
             return f"Error: Response blocked due to safety settings (Reason: {block_reason}).", usage_metadata, None

        # Assuming the first candidate is the primary one
        candidate = response.candidates[0]

        # Handle potential finish reasons other than STOP
        if candidate.finish_reason.name != "STOP":
            logging.warning(f"Generation finished with reason: {candidate.finish_reason.name}. Output might be incomplete.")
            # You might want to inspect candidate.safety_ratings here

        # Extract text content
        raw_output = "".join(part.text for part in candidate.content.parts) if candidate.content and candidate.content.parts else ""

        return raw_output, usage_metadata, prompt_feedback # Return raw text and metadata

    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}")
        # Attempt to get more details if it's a Google API error
        if hasattr(e, 'message'):
            logging.error(f"API Error Message: {e.message}")
        return f"Error: Failed to generate content. Details: {e}", None, None

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run a Gemini model based on a prompt markdown file.")
    parser.add_argument("prompt_file", help="Path to the prompt markdown file (.md).")
    parser.add_argument("-o", "--output", help="Optional path to save the output markdown file.")
    parser.add_argument("--explain", action="store_true", help="Request an explanation of the primary output (if JSON).")
    args = parser.parse_args()

    prompt_file_path = args.prompt_file
    prompt_dir = os.path.dirname(prompt_file_path) or '.' # Directory of the prompt file

    logging.info(f"Processing prompt file: {prompt_file_path}")

    # 1. Parse the prompt markdown file
    metadata, system_instructions, schema_str, prompt_template, functions_str = parse_prompt_markdown(prompt_file_path) # Added functions_str

    # Extract model and temperature, providing defaults
    model_name = metadata.get('Model', 'gemini-1.5-flash-latest') # Default to a common model
    try:
        temperature = float(metadata.get('Temperature', 0.5)) # Default temperature
    except ValueError:
        logging.warning("Invalid temperature value in metadata, using default 0.5.")
        temperature = 0.5

    # Parse schema if present
    schema = parse_schema(schema_str)
    json_mode_active = schema is not None

    # Parse functions/tools if present (Placeholder)
    # functions = parse_functions(functions_str) # Implement this if needed

    # 2. Handle INPUT_FILE directive and prepare final prompt
    input_file_match = re.search(r"INPUT_FILE:\s*(.*)", prompt_template, re.IGNORECASE | re.MULTILINE)
    input_json_content = None # Initialize
    final_prompt = prompt_template # Start with the base template

    if input_file_match:
        input_file_path_rel = input_file_match.group(1).strip()
        logging.info(f"Found INPUT_FILE directive: {input_file_path_rel}")
        input_json_content = read_file(input_file_path_rel, prompt_dir)

        # Remove the INPUT_FILE line itself from the template
        final_prompt = re.sub(r"INPUT_FILE:\s*.*\n?", "", final_prompt, flags=re.IGNORECASE | re.MULTILINE).strip()

        # Replace the placeholder if it exists, otherwise append
        placeholder = "{INPUT_data}"
        if placeholder in final_prompt:
            final_prompt = final_prompt.replace(placeholder, input_json_content)
        else:
            # If no placeholder, append the data after the main instructions
            logging.warning(f"'{placeholder}' placeholder not found in prompt. Appending input data to the end.")
            final_prompt += "\n\n" + input_json_content

    else:
        # If no INPUT_FILE, check for {INPUT_data} placeholder anyway
        if "{INPUT_data}" in final_prompt:
             logging.error("Error: Prompt contains '{INPUT_data}' placeholder but no 'INPUT_FILE:' directive was found.")
             sys.exit(1)
        # final_prompt remains as the original template content

    # --- Print Raw Input Section ---
    if input_json_content:
        print("## RAW INPUT") # Use the requested header
        print("```json") # Start a JSON code block
        try:
            # Attempt to load and re-dump the JSON for pretty printing
            parsed_input = json.loads(input_json_content)
            print(json.dumps(parsed_input, indent=2))
        except json.JSONDecodeError:
            # If it's not valid JSON (e.g., just a string), print it as is
            logging.warning("Input content is not valid JSON. Printing raw input.")
            print(input_json_content)
        except Exception as e:
             logging.warning(f"Could not pretty-print input JSON: {e}")
             print(input_json_content) # Print raw if other error occurs
        print("```") # End the JSON code block
        print("\n---\n") # Add a separator
    # --- End Print Raw Input Section ---


    # 3. Generate content using the API
    raw_output, usage_meta, prompt_fb = generate_content(
        model_name=model_name,
        prompt=final_prompt,
        temperature=temperature,
        system_instruction=system_instructions,
        schema=schema,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        # functions=functions # Placeholder
    )

    # 4. Prepare the output markdown content
    output_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    output_lines.append(f"# Gemini Output for: {os.path.basename(prompt_file_path)}")
    output_lines.append("## Request Configuration")
    output_lines.append(f"- **Model:** {model_name}")
    output_lines.append(f"- **System Instructions Provided:** {'Yes' if system_instructions else 'No'}")
    output_lines.append(f"- **Temperature:** {temperature}")
    output_lines.append(f"- **'# Controlled Output Schema' Section Found:** {'Yes' if schema_str else 'No'}")
    output_lines.append(f"- **'# Functions' Section Found:** {'Yes' if functions_str else 'No'}") # Placeholder
    output_lines.append(f"- **Function Calling Active (Tools Provided):** {'No'}") # Hardcoded No for now
    output_lines.append(f"- **JSON Output Mode Active (MIME Type):** {json_mode_active}")
    output_lines.append(f"- **Schema Parsed & Applied (for JSON Mode):** {'Yes' if json_mode_active and schema else ('No' if json_mode_active else 'N/A')}")
    output_lines.append(f"- **Safety Settings Applied:** {DEFAULT_SAFETY_SETTINGS}") # Show applied settings
    output_lines.append(f"- **Timestamp:** {timestamp}")
    output_lines.append("")

    if usage_meta:
        output_lines.append("## Usage Metadata (Primary Call)")
        output_lines.append(f"- **Prompt Token Count:** {usage_meta.prompt_token_count}")
        output_lines.append(f"- **Candidates Token Count:** {usage_meta.candidates_token_count}")
        output_lines.append(f"- **Total Token Count:** {usage_meta.total_token_count}")
        output_lines.append("")

    # Determine language for code block
    output_lang = "json" if json_mode_active and not raw_output.startswith("Error:") else ""

    output_lines.append("## RAW OUTPUT")
    output_lines.append(f"```{output_lang}")
    # Attempt to pretty-print if JSON mode was active and output isn't an error
    if json_mode_active and not raw_output.startswith("Error:"):
        try:
            parsed_output = json.loads(raw_output)
            output_lines.append(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            logging.warning("Output was expected to be JSON but failed to parse. Printing raw output.")
            output_lines.append(raw_output) # Print raw if parsing fails
        except Exception as e:
             logging.warning(f"Could not pretty-print output JSON: {e}")
             output_lines.append(raw_output) # Print raw if other error occurs
    else:
        output_lines.append(raw_output)
    output_lines.append("```")
    output_lines.append("")

    # 5. Handle Explanation Request
    explanation_output = None
    explanation_usage_meta = None
    if args.explain and json_mode_active and not raw_output.startswith("Error:"):
        logging.info("Requesting explanation for the JSON output.")
        explanation_prompt = textwrap.dedent(f"""
            Explain the following JSON data in a human-readable way. Focus on the structure, meaning of the fields, and what the data represents overall.

            JSON Data:
            ```json
            {raw_output}
            ```

            Explanation:
        """)
        # Use a default model/temp for explanation, or configure separately
        explanation_output, explanation_usage_meta, _ = generate_content(
            model_name='gemini-1.5-flash-latest', # Or use the original model
            prompt=explanation_prompt,
            temperature=0.2, # Lower temp for factual explanation
            system_instruction="You are an expert at explaining JSON data structures clearly and concisely.",
            schema=None, # No JSON mode for explanation
            safety_settings=DEFAULT_SAFETY_SETTINGS
        )
        if explanation_output and not explanation_output.startswith("Error:"):
            output_lines.append("\n---\n")
            output_lines.append("## Human-Readable Explanation")
            output_lines.append(explanation_output)
            output_lines.append("")
            if explanation_usage_meta:
                output_lines.append("## Usage Metadata (Explanation Call)")
                output_lines.append(f"- **Prompt Token Count:** {explanation_usage_meta.prompt_token_count}")
                output_lines.append(f"- **Candidates Token Count:** {explanation_usage_meta.candidates_token_count}")
                output_lines.append(f"- **Total Token Count:** {explanation_usage_meta.total_token_count}")
                output_lines.append("")


    # 6. Output the results
    final_output_content = "\n".join(output_lines)

    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_output_content)
            logging.info(f"Output successfully saved to: {args.output}")
        except Exception as e:
            logging.error(f"Error saving output file to {args.output}: {e}")
            print("\n--- Script Output ---\n") # Print to console as fallback
            print(final_output_content)
    else:
        # Print to console if no output file specified
        print(final_output_content)

if __name__ == "__main__":
    main()
