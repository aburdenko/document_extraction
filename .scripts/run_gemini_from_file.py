#!/usr/bin/env python3
# Run with python3 ./.scripts/run_gemini_from_file.py suggested-prompt-2025-06-29.md

import os
import sys
import argparse
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# --- SDK Imports ---
# This script now uses the Vertex AI SDK for generation to support integrated RAG.
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, GenerationConfig, HarmCategory, HarmBlockThreshold, SafetySetting, grounding
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, GenerationConfig, HarmCategory, HarmBlockThreshold, SafetySetting, grounding

# The following imports are for the manual RAG implementation, which is being replaced.
# They are kept here for reference but are no longer used in the primary RAG path.
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
# We need protos for schema definition and function calling
import google.ai.generativelanguage as glm
# *** FIX: Import the specific exception type ***
# --- End Add necessary imports ---

# --- Constants ---
# Default model is now primarily set in .scripts/configure.sh as GEMINI_MODEL_NAME
# A fallback is provided in the code where it's used.
OUTPUT_SUFFIX = os.getenv('OUTPUT_SUFFIX', '.md.output.md') # Get from env var, with fallback
# Regex to find sections like # System Instructions, # Prompt, etc.
SECTION_PATTERN = re.compile(r"^\s*#\s+([\w\s]+)\s*$", re.MULTILINE)
RAG_ENGINE_SECTION_KEY = "ragengine"
# --- Define the key we expect for the schema section ---
CONTROLLED_OUTPUT_SECTION_KEY = "controlled_output_schema" # Use this constant

# --- Safety Settings ---
# Define mapping from string names (used in metadata) to HarmCategory enums
HARM_CATEGORY_MAP: Dict[str, "HarmCategory"] = {
    "harassment": HarmCategory.HARM_CATEGORY_HARASSMENT,
    "hate_speech": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    "sexually_explicit": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    "dangerous_content": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
}
# Define mapping from string names (used in metadata) to HarmBlockThreshold enums
HARM_THRESHOLD_MAP: Dict[str, "HarmBlockThreshold"] = {
    "block_none": HarmBlockThreshold.BLOCK_NONE,
    "block_low_and_above": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    "block_medium_and_above": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "block_only_high": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    # Add alias for clarity/convenience
    "none": HarmBlockThreshold.BLOCK_NONE,
    "low": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "high": HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
# Default safety settings if not specified in the file
DEFAULT_SAFETY_SETTINGS: List[SafetySetting] = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
]
# --- End Safety Settings ---


# --- Logging Setup ---
# Use logging instead of print for warnings/errors for better control
# Increase level to DEBUG to see more detailed logs if needed
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)
# Example: To enable debug logs, change level=logging.INFO to level=logging.DEBUG
# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# --- Schema Conversion Function ---
def dict_to_proto_schema(schema_dict: dict) -> Optional[glm.Schema]:
    """
    Converts a Python dictionary representing a JSON schema to a glm.Schema.
    NOTE: This version has limitations and does not support $ref or complex validations.
    """
    try:
        # Type mapping dictionary for clarity
        TYPE_MAP = {
            "string": glm.Type.STRING,
            "number": glm.Type.NUMBER,
            "integer": glm.Type.INTEGER,
            "boolean": glm.Type.BOOLEAN,
            "array": glm.Type.ARRAY,
            "object": glm.Type.OBJECT,
        }

        def convert_property(prop_name, prop_details):
            if not isinstance(prop_details, dict):
                 logger.warning(f"    Expected dict for property details '{prop_name}', got {type(prop_details)}. Skipping.")
                 return None

            # Handle $ref - Basic handling: Treat as string if encountered unexpectedly
            # A proper implementation would require resolving the reference.
            if "$ref" in prop_details:
                logger.warning(f"    Schema property '{prop_name}' uses '$ref' ('{prop_details['$ref']}'). This converter has limited $ref support and will treat it as a basic type (likely STRING). The resulting schema might be incomplete.")
                # Attempt to get type from sibling keys, else default to STRING
                prop_type_str = prop_details.get("type", "string").lower()
            else:
                prop_type_str = prop_details.get("type", "string").lower()


            prop = glm.Schema(
                description=prop_details.get("description", ""),
                nullable=prop_details.get("nullable", False)
                # title is not directly mapped in glm.Schema for properties
            )

            prop.type = TYPE_MAP.get(prop_type_str, glm.Type.STRING) # Default to STRING

            if prop.type == glm.Type.STRING:
                enum = prop_details.get("enum")
                if enum and isinstance(enum, list): # Ensure enum is a list
                    prop.enum.values.extend([str(e) for e in enum]) # Ensure values are strings
                # pattern is not directly mapped in glm.Schema

            elif prop.type == glm.Type.ARRAY:
                items = prop_details.get("items")
                if items:
                    # Handle case where items is a schema object itself
                    if isinstance(items, dict):
                        item_schema = convert_property("items", items) # Recursive call
                        if item_schema:
                            prop.items = item_schema
                        else:
                            # If recursive call failed, default to STRING array
                            logger.warning(f"    Could not determine schema for array items in '{prop_name}' (recursive conversion failed). Defaulting to STRING array.")
                            prop.items = glm.Schema(type=glm.Type.STRING)
                    else:
                        logger.warning(f"    Array property '{prop_name}' has 'items' field but it's not a dictionary. Defaulting to STRING array.")
                        prop.items = glm.Schema(type=glm.Type.STRING)
                else:
                     logger.warning(f"    Array property '{prop_name}' missing 'items' definition. Defaulting to STRING array.")
                     prop.items = glm.Schema(type=glm.Type.STRING) # Default items to STRING if not specified


            elif prop.type == glm.Type.OBJECT:
                properties = prop_details.get("properties")
                if properties and isinstance(properties, dict):
                    for name, details in properties.items():
                        converted_sub_prop = convert_property(name, details) # Recursive call
                        if converted_sub_prop:
                             prop.properties[name] = converted_sub_prop
                required = prop_details.get("required")
                if required and isinstance(required, list):
                    prop.required.extend(required)
                # propertyOrdering is ignored

            # Ignore other keys like title, $defs (at property level)

            return prop

        # Start conversion from the root of the schema dictionary
        # Need to handle $defs at the root if present
        if "$defs" in schema_dict:
             logger.warning("    Root schema contains '$defs'. This converter does not resolve references within '$defs'. Schema might be incomplete.")
             # We proceed, but $ref resolution won't happen.

        root_schema = convert_property("root", schema_dict)
        return root_schema

    except Exception as e:
        logger.error(f"    Error converting dictionary to proto schema: {e}", exc_info=True) # Log traceback
        return None
# --- End Schema Conversion Function ---

# --- Function Declaration Conversion ---
def dict_list_to_proto_tool(func_list_dict: List[Dict[str, Any]]) -> Optional[glm.Tool]:
    """Converts a list of function definition dictionaries to a glm.Tool."""
    if not isinstance(func_list_dict, list):
        logger.error(f"    Expected a list for function definitions, got {type(func_list_dict)}.")
        return None

    function_declarations = []
    for i, func_dict in enumerate(func_list_dict):
        if not isinstance(func_dict, dict):
            logger.warning(f"    Skipping item at index {i} in function list: Expected a dictionary, got {type(func_dict)}.")
            continue

        name = func_dict.get("name")
        description = func_dict.get("description")
        parameters_dict = func_dict.get("parameters")

        if not name:
            logger.warning(f"    Skipping function definition at index {i}: Missing 'name'.")
            continue

        # Parameters are optional, but if provided, they must be a valid schema dict
        parameters_schema = None
        if parameters_dict:
            if isinstance(parameters_dict, dict):
                # We need to ensure the parameters dict itself follows JSON schema structure,
                # especially the top level 'type' which should usually be 'object'.
                if 'type' not in parameters_dict:
                     logger.warning(f"    Function '{name}' parameters missing 'type'. Assuming 'object'.")
                     parameters_dict['type'] = 'object' # Assume object if not specified
                if parameters_dict.get('type') != 'object':
                     logger.warning(f"    Function '{name}' parameters 'type' is '{parameters_dict.get('type')}', expected 'object'. Conversion might be unexpected.")

                parameters_schema = dict_to_proto_schema(parameters_dict)
                if not parameters_schema:
                    logger.warning(f"    Could not convert parameters for function '{name}' to proto schema. Skipping parameters.")
            else:
                logger.warning(f"    Function '{name}' parameters field is not a dictionary. Skipping parameters.")

        # Create the FunctionDeclaration proto
        func_decl = glm.FunctionDeclaration(
            name=name,
            description=description or "", # Description is optional
        )
        # Only add parameters if they were successfully converted
        if parameters_schema:
            func_decl.parameters = parameters_schema

        function_declarations.append(func_decl)
        logger.info(f"    Parsed function declaration: '{name}'")


    if not function_declarations:
        logger.warning("    No valid function declarations found in the provided list.")
        return None

    # Wrap the declarations in a Tool
    tool = glm.Tool(function_declarations=function_declarations)
    return tool

# --- End Function Declaration Conversion ---


# --- Enhanced Metadata Parsing ---
def parse_metadata_and_body(file_content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parses specific metadata keys found anywhere in the file content
    and returns them as a dictionary. Handles type conversions.
    Returns the full original content as the body for further section parsing.
    """
    metadata_dict: Dict[str, Any] = {}
    body_content = file_content.strip() # Keep original content for section parsing

    # --- Metadata Extraction Functions ---
    def find_value(key_pattern: str, content: str) -> Optional[str]:
        match = re.search(rf"^\s*{key_pattern}:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else None

    def find_float(key_pattern: str, content: str) -> Optional[float]:
        value_str = find_value(key_pattern, content)
        if value_str:
            try:
                return float(value_str)
            except ValueError:
                logger.warning(f"    Could not parse float value for '{key_pattern}': {value_str}")
        return None

    def find_int(key_pattern: str, content: str) -> Optional[int]:
        value_str = find_value(key_pattern, content)
        if value_str:
            try:
                # Allow float strings like "100.0" to be parsed as int
                return int(float(value_str))
            except ValueError:
                logger.warning(f"    Could not parse integer value for '{key_pattern}': {value_str}")
        return None

    def find_string_list(key_pattern: str, content: str) -> Optional[List[str]]:
        value_str = find_value(key_pattern, content)
        if value_str:
            # Assume comma-separated, trim whitespace
            return [item.strip() for item in value_str.split(',') if item.strip()]
        return None

    def find_safety_settings(content: str) -> Optional[List[SafetySetting]]:
        """Parses safety settings like 'Safety: harassment=none, hate_speech=low'"""
        value_str = find_value("Safety(?: Settings)?", content)
        if not value_str:
            return None

        settings: List[SafetySetting] = []
        settings: List[SafetySetting] = []
        pairs = [item.strip() for item in value_str.split(',') if item.strip()]
        for pair in pairs:
            try:
                category_str, threshold_str = [p.strip().lower().replace('-', '_') for p in pair.split('=')]
                category = HARM_CATEGORY_MAP.get(category_str)
                threshold = HARM_THRESHOLD_MAP.get(threshold_str)

                if category and threshold:
                    settings.append(SafetySetting(category=category, threshold=threshold))
                else:
                    if not category:
                        logger.warning(f"    Invalid safety category '{category_str}' found in metadata. Skipping.")
                    if not threshold:
                        logger.warning(f"    Invalid safety threshold '{threshold_str}' found in metadata. Skipping.")
            except ValueError:
                logger.warning(f"    Could not parse safety setting pair: '{pair}'. Expected format 'category=threshold'. Skipping.")
            except Exception as e:
                 logger.warning(f"    Error parsing safety setting pair '{pair}': {e}. Skipping.")

        return settings if settings else None
    # --- End Metadata Extraction Functions ---


    # --- Parse Known Metadata ---
    metadata_dict['model_name'] = find_value(r"Model(?: Used)?(?: \(intended\))?", file_content)
    metadata_dict['temperature'] = find_float(r"Temperature", file_content)
    metadata_dict['top_p'] = find_float(r"Top P", file_content)
    metadata_dict['top_k'] = find_int(r"Top K", file_content)
    metadata_dict['max_output_tokens'] = find_int(r"Max(?: Output)? Tokens", file_content)
    metadata_dict['stop_sequences'] = find_string_list(r"Stop Sequences", file_content)
    metadata_dict['safety_settings'] = find_safety_settings(file_content)
    # --- End Parse Known Metadata ---

    # Filter out None values
    metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}

    return metadata_dict, body_content


# --- Modified parse_sections Function ---
# *** FIX: Update return type hint ***
def parse_sections(text_content: str) -> Tuple[Dict[str, str], Optional[glm.Schema], bool, Optional[glm.Tool], bool, Optional[str]]:
    """
    Parses the text content into sections based on headings.
    Checks for '# Controlled Output Schema', '# Functions', and '# RagEngine'.
    Returns:
        - sections: Dictionary of section names to content.
        - proto_schema: Parsed proto schema from '# Controlled Output Schema' (or None if parsing fails).
        - controlled_output_section_found: Flag indicating if '# Controlled Output Schema' was found.
        - proto_tool: Parsed proto tool from '# Functions' (or None).
        - functions_section_found: Flag indicating if '# Functions' was found.
        - rag_engine_endpoint: The display name of the Vector Search endpoint from '# RagEngine' section.
    """
    sections: Dict[str, str] = {}
    last_pos = 0
    current_section_name = "initial_content" # Content before first heading

    for match in SECTION_PATTERN.finditer(text_content):
        # Convert heading to key: lowercase, replace spaces with underscores
        section_name = match.group(1).strip().lower().replace(" ", "_")
        start, end = match.span()
        section_content = text_content[last_pos:start].strip()
        if section_content or current_section_name == "initial_content": # Keep initial even if empty
            sections[current_section_name] = section_content

        current_section_name = section_name
        last_pos = end

    section_content = text_content[last_pos:].strip()
    if section_content:
        sections[current_section_name] = section_content

    # --- Schema Extraction ---
    schema_dict = None
    proto_schema = None # Initialize to None
    controlled_output_section_found = CONTROLLED_OUTPUT_SECTION_KEY in sections

    if controlled_output_section_found:
        logger.info(f"    Found '# {CONTROLLED_OUTPUT_SECTION_KEY.replace('_', ' ').title()}' section.")
        schema_json_str = sections[CONTROLLED_OUTPUT_SECTION_KEY]
        # Remove potential ```json ``` wrappers
        schema_json_str = re.sub(r"^\s*```(?:json)?\s*", "", schema_json_str, flags=re.IGNORECASE | re.MULTILINE)
        schema_json_str = re.sub(r"\s*```\s*$", "", schema_json_str, flags=re.MULTILINE)
        schema_json_str = schema_json_str.strip()

        if schema_json_str:
            try:
                schema_dict = json.loads(schema_json_str)
                logger.info("    JSON schema content parsed successfully.")
                # --- FIX: Manually resolve the specific $ref in the schema ---
                # The dict_to_proto_schema converter doesn't support $ref. We can resolve it here
                # after parsing the JSON and before passing it to the converter.
                try:
                    # Define the object we are referencing
                    ref_target = schema_dict["properties"]["prescriptions"]["items"]["properties"]["bilNotesSummary"]["items"]
                    # Find the location of the reference and replace it with the actual object
                    schema_dict["properties"]["prescriptions"]["items"]["properties"]["genNotesSummary"]["items"] = ref_target
                    logger.info("    Manually resolved '$ref' for 'genNotesSummary.items'.")
                except KeyError as e:
                    logger.warning(f"    Could not manually resolve $ref, path not found in schema: {e}")
                # --- END FIX ---
                # Attempt conversion
                proto_schema = dict_to_proto_schema(schema_dict) # Assign result here
                if proto_schema:
                    logger.info("    Successfully converted schema to proto format (basic validation).")
                else:
                     # This path is taken if dict_to_proto_schema explicitly returns None (e.g., due to exception)
                     logger.warning("    Conversion to proto schema failed (returned None). Check previous errors.")
            except json.JSONDecodeError as e:
                logger.warning(f"    Failed to parse JSON content in '# {CONTROLLED_OUTPUT_SECTION_KEY.replace('_', ' ').title()}' section: {e}")
                # proto_schema remains None
            except Exception as e:
                logger.warning(f"    Error processing schema from '# {CONTROLLED_OUTPUT_SECTION_KEY.replace('_', ' ').title()}' section: {e}", exc_info=True)
                # proto_schema remains None
        else:
             logger.warning(f"    '# {CONTROLLED_OUTPUT_SECTION_KEY.replace('_', ' ').title()}' section is empty.")
             # proto_schema remains None

    # --- Function Declaration Extraction ---
    proto_tool = None
    # *** FIX: Store the result of the check ***
    functions_section_found = 'functions' in sections # This key is derived from "# Functions"

    if functions_section_found:
        logger.info("    Found '# Functions' section.")
        functions_json_str = sections['functions']
        # Remove potential ```json ``` wrappers
        functions_json_str = re.sub(r"^\s*```(?:json)?\s*", "", functions_json_str, flags=re.IGNORECASE | re.MULTILINE)
        functions_json_str = re.sub(r"\s*```\s*$", "", functions_json_str, flags=re.MULTILINE)
        functions_json_str = functions_json_str.strip()

        if functions_json_str:
            try:
                functions_list = json.loads(functions_json_str)
                logger.info("    Function definitions JSON parsed successfully.")
                proto_tool = dict_list_to_proto_tool(functions_list)
                if proto_tool:
                    logger.info("    Successfully converted function definitions to proto Tool format.")
                else:
                    logger.warning("    Failed to convert parsed JSON list to proto Tool (see previous errors).")
            except json.JSONDecodeError as e:
                logger.warning(f"    Failed to parse JSON content in '# Functions' section: {e}")
            except Exception as e:
                logger.warning(f"    Error processing functions from '# Functions' section: {e}", exc_info=True)
        else:
            logger.warning("    '# Functions' section is empty.")

    # --- RAG Engine Extraction ---
    rag_engine_endpoint = sections.get(RAG_ENGINE_SECTION_KEY)
    if rag_engine_endpoint:
        rag_engine_endpoint = rag_engine_endpoint.strip()
        logger.info(f"    Found '# RagEngine' section. RAG resource specified: '{rag_engine_endpoint}'")


    # *** FIX: Return the functions_section_found boolean ***
    return sections, proto_schema, controlled_output_section_found, proto_tool, functions_section_found, rag_engine_endpoint
# --- End Modified parse_sections Function ---


def call_gemini_with_prompt_file(prompt_filepath: str):
    """Processes a single prompt file and calls the Gemini API."""
    try:
        logger.info(f"--- Starting processing for {Path(prompt_filepath).name} ---")
        logger.info(f"--- Starting processing for {Path(prompt_filepath).name} ---")
        filepath = Path(prompt_filepath)
        print(f"[{datetime.now()}] Processing prompt from: {filepath.name}") # Use print for top-level status

        file_content = filepath.read_text()

        # 1. Parse Metadata and Body
        metadata, body = parse_metadata_and_body(file_content)
        logger.info(f"    Parsed Metadata: {metadata}")

        # 2. Parse Sections, Schema, and Functions
        # *** FIX: Unpack the new return value ***
        sections, proto_schema, controlled_output_section_found, proto_tool, functions_section_found, rag_engine_endpoint = parse_sections(body)

        system_instructions = sections.get("system_instructions")
        user_prompt = sections.get("prompt")

        if not user_prompt:
            # Fallback logic for finding the prompt (same as before)
            if 'initial_content' in sections and not system_instructions and sections['initial_content']:
                 logger.warning("    No '# Prompt' section found. Using initial content before first heading as prompt.")
                 user_prompt = sections.get('initial_content')
            elif 'initial_content' in sections and system_instructions and sections['initial_content']:
                 logger.warning("    No '# Prompt' section found. Using content between '# System Instructions' and next heading (or EOF) as prompt.")
                 user_prompt = sections.get('initial_content')
            else:
                 logger.error("    Could not find a '# Prompt' section or suitable fallback content.")
                 output_filename = filepath.with_suffix(OUTPUT_SUFFIX)
                 output_filename.write_text(f"# Gemini Output for: {filepath.name}\n\n---\n\nPROCESSING ERROR\nDetails: No '# Prompt' section found and no fallback content available.")
                 return


        # --- RAG Engine Logic ---
        # This new approach uses Vertex AI's integrated RAG.
        # It creates a tool from the specified RAG corpus and passes it to the model.
        # The model then handles the retrieval and grounding automatically.
        rag_tool = None
        if rag_engine_endpoint:
            logger.info("--- RAG Engine Processing ---")
            try:
                project_id = os.getenv("PROJECT_ID")
                region = os.getenv("REGION", "us-central1")
                if not project_id or not region:
                    raise ValueError("PROJECT_ID and REGION must be set for RAG Engine.")
                region = os.getenv("REGION", "us-central1")
                if not project_id or not region:
                    raise ValueError("PROJECT_ID and REGION must be set for RAG Engine.")

                # Initialize Vertex AI SDK (if not already done)
                vertexai.init(project=project_id, location=region)
                vertexai.init(project=project_id, location=region)

                rag_source = None
                rag_resource_string = rag_engine_endpoint.strip()

                # The RAG service now supports passing the corpus resource name directly
                # Case 1: It's a full RagCorpus resource name
                if "/ragCorpora/" in rag_resource_string:
                    logger.info(f"    Interpreting '{rag_resource_string}' as a RagCorpus resource name.")
                    rag_source = grounding.RagSource(rag_corpora=[rag_resource_string])

                # Case 2: It's a full Vector Search Index resource name
                elif "/indexes/" in rag_resource_string:
                    logger.info(f"    Interpreting '{rag_resource_string}' as a Vector Search Index resource name.")
                    rag_source = grounding.RagSource(vector_search_index=rag_resource_string)

                # Case 3: It's a display name for an Endpoint or Index
                else:
                    logger.info(f"    Interpreting '{rag_resource_string}' as a display name. Searching for a matching Vector Search Endpoint in region '{region}'...")
                    aiplatform.init(project=project_id, location=region)
                    endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
                        filter=f'display_name="{rag_resource_string}"'
                    )

                    if endpoints:
                        endpoint = endpoints[0]
                        if len(endpoints) > 1:
                            logger.warning(f"    Found multiple endpoints with the same name. Using the first one: {endpoint.resource_name}")
                        logger.info(f"    Found endpoint: {endpoint.resource_name}")
                        if not endpoint.deployed_indexes:
                            raise ValueError(f"Endpoint '{endpoint.resource_name}' has no deployed indexes.")
                        index_resource_name = endpoint.deployed_indexes[0].index
                        logger.info(f"    Using underlying index: {index_resource_name}")
                        rag_source = grounding.RagSource(vector_search_index=index_resource_name)
                    else:
                        # If no endpoint is found, maybe it's an index display name
                        logger.info(f"    No endpoint found. Searching for a matching Vector Search Index with display name '{rag_resource_string}'...")
                        indexes = aiplatform.MatchingEngineIndex.list(
                            filter=f'display_name="{rag_resource_string}"'
                        )
                        if indexes:
                            index = indexes[0]
                            if len(indexes) > 1:
                                logger.warning(f"    Found multiple indexes with the same name. Using the first one: {index.resource_name}")
                            logger.info(f"    Found index: {index.resource_name}")
                            rag_source = grounding.RagSource(vector_search_index=index.resource_name)
                        else:
                             raise ValueError(f"Could not find a RagCorpus, Vector Search Endpoint, or Vector Search Index matching '{rag_resource_string}' in region '{region}'.")

                if rag_source:
                    logger.info("    Creating RAG retrieval tool...")
                    retrieval = Tool.from_retrieval(
                        grounding.Retrieval(source=rag_source)
                    )
                    rag_tool = retrieval
            except Exception as e:
                logger.error(f"    Error during RAG processing: {e}", exc_info=True)
            logger.info("--- End RAG Engine Processing ---")

        logger.info(f"    System Instructions Provided: {'Yes' if system_instructions else 'No'}")
        logger.info(f"    User Prompt (first 50 chars): '{user_prompt[:50]}...'")
        logger.info(f"    Function Declarations Provided: {'Yes' if proto_tool else 'No'}")

        # 3. Determine Model and Generation Config Parameters
        model_name = metadata.get('model_name', os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')) # Prioritizes metadata, then env var, then fallback
        model_name = metadata.get('model_name', os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')) # Prioritizes metadata, then env var, then fallback
        logger.info(f"    Using Model: {model_name}")

        generation_config_args: Dict[str, Any] = {}
        # Populate from metadata if present
        if 'temperature' in metadata:
            generation_config_args['temperature'] = metadata['temperature']
            logger.info(f"    Temperature: {metadata['temperature']}")
        if 'top_p' in metadata:
            generation_config_args['top_p'] = metadata['top_p']
            logger.info(f"    Top P: {metadata['top_p']}")
        if 'top_k' in metadata:
            generation_config_args['top_k'] = metadata['top_k']
            logger.info(f"    Top K: {metadata['top_k']}")
        if 'max_output_tokens' in metadata:
            generation_config_args['max_output_tokens'] = metadata['max_output_tokens']
            logger.info(f"    Max Output Tokens: {metadata['max_output_tokens']}")
        if 'stop_sequences' in metadata:
            generation_config_args['stop_sequences'] = metadata['stop_sequences']
            logger.info(f"    Stop Sequences: {metadata['stop_sequences']}")

        # Configure JSON mode *only* if requested AND function calling is NOT active
        activate_json_mode = controlled_output_section_found and not proto_tool
        if proto_tool and controlled_output_section_found:
            logger.warning(f"    Both '# Functions' and '# {CONTROLLED_OUTPUT_SECTION_KEY.replace('_', ' ').title()}' sections found. Function calling takes precedence; ignoring '# {CONTROLLED_OUTPUT_SECTION_KEY.replace('_', ' ').title()}' for response_mime_type setting.")

        logger.info(f"    JSON Output Mode Active (via mime_type): {activate_json_mode}")

        if activate_json_mode:
            logger.info("    Configuring model for JSON output (mime type: application/json).")
            generation_config_args['response_mime_type'] = "application/json"
            # Check if proto_schema is a valid object (not None)
            if proto_schema:
                logger.info("    Applying parsed schema to generation config.")
                generation_config_args['response_schema'] = proto_schema
            else:
                # This log indicates why the schema wasn't applied
                logger.warning("    JSON mode activated, but no valid schema was parsed/converted. Requesting generic JSON.")

        generation_config = GenerationConfig(**generation_config_args) if generation_config_args else None

        # Determine Safety Settings
        safety_settings = metadata.get('safety_settings', DEFAULT_SAFETY_SETTINGS)
        logger.info(f"    Using Safety Settings: {safety_settings}")

        # 4. Configure Credentials
        # The Vertex AI SDK primarily uses Application Default Credentials (ADC).
        # The `vertexai.init()` call handles this. We'll check for the env var for logging.
        google_creds_env = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if google_creds_env:
             logger.info(f"    Using GOOGLE_APPLICATION_CREDENTIALS: {google_creds_env}")
        else:
            logger.info("    Using Application Default Credentials (ADC) from the environment.")


        # 5. Prepare Model
        model = GenerativeModel(
            model_name=model_name,
            system_instruction=system_instructions,
            safety_settings=safety_settings # Apply safety settings here
        )

        # 6. Call Gemini API (Primary Call)
        logger.info("    Calling Gemini API (Primary Call)...")
        # Combine function calling tools and the RAG tool
        all_tools = []
        if proto_tool:
            all_tools.append(proto_tool)
        if rag_tool:
            all_tools.append(rag_tool)

        response = model.generate_content(
            user_prompt,
            generation_config=generation_config,
            tools=all_tools if all_tools else None
            # Safety settings moved to model initialization
        )
        logger.info("    Primary API call complete.")

        # 7. Process and Prepare Output Content
        output_filename = filepath.with_suffix(OUTPUT_SUFFIX)
        output_content = f"# Gemini Output for: {filepath.name}\n"
        output_content += f"## Request Configuration\n"
        output_content += f"- **Model:** {model_name}\n"
        output_content += f"- **System Instructions Provided:** {'Yes' if system_instructions else 'No'}\n"
        # Add details from generation config if used
        if generation_config:
             if generation_config.temperature is not None:
                  output_content += f"- **Temperature:** {generation_config.temperature}\n"
             if generation_config.top_p is not None:
                  output_content += f"- **Top P:** {generation_config.top_p}\n"
             if generation_config.top_k is not None:
                  output_content += f"- **Top K:** {generation_config.top_k}\n"
             if generation_config.max_output_tokens is not None:
                  output_content += f"- **Max Output Tokens:** {generation_config.max_output_tokens}\n"
             if generation_config.stop_sequences:
                  output_content += f"- **Stop Sequences:** {generation_config.stop_sequences}\n"

        # Reflect function/JSON mode status accurately
        output_content += f"- **'# RagEngine' Section Found:** {'Yes' if rag_engine_endpoint else 'No'}\n"
        output_content += f"- **RAG Tool Provided to Model:** {'Yes' if rag_tool else 'No'}\n"
        output_content += f"- **'# {CONTROLLED_OUTPUT_SECTION_KEY.replace('_', ' ').title()}' Section Found:** {'Yes' if controlled_output_section_found else 'No'}\n"
        # *** FIX: Use the correct variable here ***
        output_content += f"- **'# Functions' Section Found:** {'Yes' if functions_section_found else 'No'}\n"
        output_content += f"- **Function Calling Active (Tools Provided):** {'Yes' if proto_tool else 'No'}\n"
        output_content += f"- **JSON Output Mode Active (MIME Type):** {activate_json_mode}\n"
        # Clarify schema status: Was it found? Was it successfully parsed and applied?
        output_content += f"- **Schema Parsed & Applied (for JSON Mode):** {'Yes' if activate_json_mode and proto_schema else 'No'}\n"

        # Add safety settings used
        output_content += f"- **Safety Settings Applied:** {safety_settings}\n"
        output_content += f"- **Timestamp:** {datetime.now()}\n\n"

        # Add Usage Metadata if available from primary call
        usage_metadata = getattr(response, 'usage_metadata', None)
        if usage_metadata:
            output_content += f"## Usage Metadata (Primary Call)\n"
            output_content += f"- **Prompt Token Count:** {getattr(usage_metadata, 'prompt_token_count', 'N/A')}\n"
            output_content += f"- **Candidates Token Count:** {getattr(usage_metadata, 'candidates_token_count', 'N/A')}\n"
            # Function call usage might be counted differently or included here
            output_content += f"- **Total Token Count:** {getattr(usage_metadata, 'total_token_count', 'N/A')}\n\n"


        # --- Process Response Content (Text or Function Call) ---
        # Check for grounding metadata to display the retrieved context
        try:
            grounding_metadata = getattr(response.candidates[0], 'grounding_metadata', None)
        except IndexError:
            grounding_metadata = None # Handle cases with no candidates

        if grounding_metadata and hasattr(grounding_metadata, 'retrieval_queries'):
            retrieved_context = ""
            for query in getattr(grounding_metadata, 'retrieval_queries', []):
                for chunk in getattr(query, 'retrieved_chunks', []):
                     # The source attribute contains the GCS URI
                     source_uri = getattr(chunk, 'source', 'N/A')
                     content = getattr(chunk, 'content', 'N/A')
                     retrieved_context += f"Source: {source_uri}\n"
                     retrieved_context += f"Content: {content}\n---\n"
            if retrieved_context:
                 output_content += f"## RAG CONTEXT\n\n"
                 output_content += f"```text\n{retrieved_context}\n```\n\n"

        # Check for grounding metadata to display the retrieved context
        try:
            grounding_metadata = getattr(response.candidates[0], 'grounding_metadata', None)
        except IndexError:
            grounding_metadata = None # Handle cases with no candidates

        if grounding_metadata and hasattr(grounding_metadata, 'retrieval_queries'):
            retrieved_context = ""
            for query in getattr(grounding_metadata, 'retrieval_queries', []):
                for chunk in getattr(query, 'retrieved_chunks', []):
                     # The source attribute contains the GCS URI
                     source_uri = getattr(chunk, 'source', 'N/A')
                     content = getattr(chunk, 'content', 'N/A')
                     retrieved_context += f"Source: {source_uri}\n"
                     retrieved_context += f"Content: {content}\n---\n"
            if retrieved_context:
                 output_content += f"## RAG CONTEXT\n\n"
                 output_content += f"```text\n{retrieved_context}\n```\n\n"

        output_content += f"## RAW OUTPUT\n\n"
        function_call_requested = False
        response_text = ""
        raw_json_output_for_explanation = None # Store successfully parsed JSON here

        try:
            # Check for function call first
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        fc = part.function_call
                        logger.info(f"    Model requested function call: {fc.name}")
                        output_content += f"**Function Call Requested:**\n"
                        output_content += f"- **Name:** `{fc.name}`\n"
                        # Pretty print args if they exist
                        args_dict = type(fc.args).to_dict(fc.args) # Convert proto Struct to dict
                        if args_dict:
                             pretty_args = json.dumps(args_dict, indent=2, ensure_ascii=False, sort_keys=True)
                             output_content += f"- **Arguments:**\n```json\n{pretty_args}\n```\n"
                        else:
                             output_content += f"- **Arguments:** (None provided)\n"
                        function_call_requested = True
                        break # Stop after finding the first function call

            # If no function call was requested, try to get text
            if not function_call_requested:
                try:
                    response_text = response.text
                    logger.info("    Model returned text content.")
                except ValueError as e:
                    # Handle cases where accessing response.text fails (e.g., blocked content)
                    logger.warning(f"    Could not access response text directly (may be blocked or empty): {e}")
                    response_text = "" # Ensure response_text is empty string
                except Exception as e:
                     logger.error(f"    Unexpected error accessing response text: {e}")
                     response_text = f"Error accessing response text: {e}"

        except Exception as e:
            logger.error(f"    Error processing response parts/text: {e}", exc_info=True)
            output_content += f"Error processing response content: {e}\n"


        # --- Format Text Output (if applicable) ---
        if not function_call_requested and response_text:
            # Attempt to pretty-print if it looks like JSON, even if JSON mode wasn't forced
            is_likely_json = response_text.strip().startswith('{') and response_text.strip().endswith('}') \
                          or response_text.strip().startswith('[') and response_text.strip().endswith(']')
            parsed_successfully = False
            pretty_json_output = ""

            if is_likely_json:
                try:
                    # Attempt 1: Direct parse
                    logger.info("    Attempting direct JSON parsing of text response...")
                    parsed_json = json.loads(response_text)
                    pretty_json_output = json.dumps(parsed_json, indent=2, ensure_ascii=False, sort_keys=True)
                    output_content += f"```json\n{pretty_json_output}\n```\n" # Wrap in json code block
                    parsed_successfully = True
                    logger.info("    Successfully parsed and pretty-printed JSON text response (Attempt 1).")
                except json.JSONDecodeError as e1:
                    logger.warning(f"    Direct JSON parsing failed: {e1}. Checking for double encoding...")
                    # Attempt 2: Double parse (string literal containing JSON)
                    is_string_literal = response_text.startswith('"') and response_text.endswith('"')
                    if is_string_literal:
                        logger.info("    Response appears to be a string literal. Attempting double parsing...")
                        try:
                            decoded_string = json.loads(response_text)
                            if isinstance(decoded_string, str):
                                logger.info("    Result is a string. Attempting to parse inner JSON...")
                                parsed_inner_json = json.loads(decoded_string)
                                pretty_json_output = json.dumps(parsed_inner_json, indent=2, ensure_ascii=False, sort_keys=True)
                                output_content += f"```json\n{pretty_json_output}\n```\n" # Wrap in json code block
                                parsed_successfully = True
                                logger.info("    Successfully parsed and pretty-printed double-encoded JSON string (Attempt 2).")
                            else:
                                logger.warning(f"    Decoded string literal resulted in non-string type '{type(decoded_string)}'. Treating as parse failure.")
                        except json.JSONDecodeError as e2:
                            logger.warning(f"    Parsing the inner JSON string failed: {e2}.")
                        except Exception as e_inner:
                            logger.warning(f"    Error during inner JSON parsing attempt: {e_inner}.")
                    else:
                        logger.warning("    Response text is not a string literal containing JSON.")

            # Fallback: If not likely JSON or parsing failed, save raw text.
            if not parsed_successfully:
                if is_likely_json: # Add warning if we expected JSON but failed
                     logger.warning("    Could not parse model output as valid JSON. Saving raw text.")
                # Keep fences for the raw text fallback for clarity
                output_content += f"```text\n{response_text}\n```\n"
            elif activate_json_mode: # Store the pretty JSON if parsing succeeded AND JSON mode was active
                raw_json_output_for_explanation = pretty_json_output


        elif not function_call_requested and not response_text:
             # Handle cases with no text output and no function call (e.g., blocked by safety)
             try:
                  finish_reason = "N/A"
                  safety_ratings_str = "N/A"
                  if response.candidates:
                      candidate = response.candidates[0]
                      finish_reason = getattr(candidate.finish_reason, 'name', 'UNKNOWN')
                      safety_ratings = getattr(candidate, 'safety_ratings', [])
                      safety_ratings_str = ', '.join([f"{r.category.name}={r.probability.name}" for r in safety_ratings]) if safety_ratings else "None"

                  output_content += f"(No text content or function call in response. Finish Reason: {finish_reason}, Safety Ratings: [{safety_ratings_str}])"
                  # Also log the prompt feedback if available
                  prompt_feedback = getattr(response, 'prompt_feedback', None)
                  if prompt_feedback:
                      block_reason = getattr(prompt_feedback, 'block_reason', None)
                      block_reason_msg = getattr(prompt_feedback, 'block_reason_message', '')
                      pf_safety_ratings = getattr(prompt_feedback, 'safety_ratings', [])
                      pf_safety_ratings_str = ', '.join([f"{r.category.name}={r.probability.name}" for r in pf_safety_ratings]) if pf_safety_ratings else "None"
                      logger.warning(f"    Prompt Feedback: BlockReason={block_reason}, Message='{block_reason_msg}', SafetyRatings=[{pf_safety_ratings_str}]")
                      output_content += f"\nPrompt Feedback: BlockReason={block_reason}, Message='{block_reason_msg}', SafetyRatings=[{pf_safety_ratings_str}]"

             except Exception as e:
                  # Fallback if response structure is unexpected
                  logger.error(f"    Error extracting details from empty/blocked response: {e}", exc_info=True)
                  output_content += f"(No text content or function call in response. Response details might be incomplete: {response})"


        # --- 8. Make Second Call for Human-Readable Explanation (if applicable) ---
        # Check if JSON mode was active AND if a schema object was successfully created
        if activate_json_mode and raw_json_output_for_explanation and proto_schema:
            logger.info("    Controlled output JSON generated *with schema*. Making second API call for human-readable explanation...")
            explanation_text = ""
            explanation_usage_metadata = None
            try:
                # Use the same model instance, default generation config (no JSON mode, schema, etc.)
                explanation_prompt = (
                    f"The following JSON data was generated based on a specific request and schema.\n" # Mention schema
                    f"Please explain this JSON data in a clear, human-readable format. Focus on the meaning, structure, and key information contained within it, considering the constraints imposed by the original schema.\n\n"
                    f"```json\n{raw_json_output_for_explanation}\n```"
                )
                explanation_response: GenerateContentResponse = model.generate_content(
                    explanation_prompt
                    # Use default safety settings from the model
                    # Use default generation config (no temp, top_k etc specified here)
                )
                explanation_text = explanation_response.text
                explanation_usage_metadata = getattr(explanation_response, 'usage_metadata', None)
                logger.info("    Explanation call successful.")
                output_content += f"\n\n## Human-Readable Explanation\n\n{explanation_text}\n" # No code block needed for explanation

                # Add usage metadata for the second call
                if explanation_usage_metadata:
                    output_content += f"\n\n## Usage Metadata (Explanation Call)\n"
                    output_content += f"- **Prompt Token Count:** {getattr(explanation_usage_metadata, 'prompt_token_count', 'N/A')}\n"
                    output_content += f"- **Candidates Token Count:** {getattr(explanation_usage_metadata, 'candidates_token_count', 'N/A')}\n"
                    output_content += f"- **Total Token Count:** {getattr(explanation_usage_metadata, 'total_token_count', 'N/A')}\n"

            except Exception as e:
                logger.warning(f"    Failed to get human-readable explanation from second API call: {e}", exc_info=True)
                output_content += f"\n\n## Human-Readable Explanation\n\n(Failed to generate explanation: {e})\n"
        elif activate_json_mode and raw_json_output_for_explanation and not proto_schema:
             logger.info("    Controlled output JSON generated *without a successfully parsed schema*. Skipping explanation call.")
             # Optionally add a note in the output file
             # output_content += f"\n\n## Human-Readable Explanation\n\n(Skipped: JSON generated without a successfully parsed schema being applied.)\n"


        # --- 9. Save Final Output ---
        output_filename.write_text(output_content)
        logger.info(f"--- Finished processing for {filepath.name} ---")
        logger.info(f"--- Finished processing for {filepath.name} ---")
        print(f"    Output saved to: {output_filename}") # Use print for final status

    except FileNotFoundError:
        logger.error(f"Error processing '{prompt_filepath}': File not found.")
    except Exception as e:
        # Enhanced error logging to provide a full traceback in the console
        logger.error(f"An unexpected error occurred during processing of '{prompt_filepath}'.", exc_info=True)
        output_filename = Path(prompt_filepath).with_suffix(OUTPUT_SUFFIX)
        # Write a more informative error message to the output file
        error_content = (
            f"# Gemini Output for: {Path(prompt_filepath).name}\n\n"
            f"---\n\n"
            f"FATAL PROCESSING ERROR\n"
            f"Type: {type(e).__name__}\n"
            f"Details: {e}\n\n"
            f"Please check the application logs for a full traceback."
        )
        output_filename.write_text(error_content)
        logger.info(f"Error details saved to {output_filename}")
        # Enhanced error logging to provide a full traceback in the console
        logger.error(f"An unexpected error occurred during processing of '{prompt_filepath}'.", exc_info=True)
        output_filename = Path(prompt_filepath).with_suffix(OUTPUT_SUFFIX)
        # Write a more informative error message to the output file
        error_content = (
            f"# Gemini Output for: {Path(prompt_filepath).name}\n\n"
            f"---\n\n"
            f"FATAL PROCESSING ERROR\n"
            f"Type: {type(e).__name__}\n"
            f"Details: {e}\n\n"
            f"Please check the application logs for a full traceback."
        )
        output_filename.write_text(error_content)
        logger.info(f"Error details saved to {output_filename}")


def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Process prompt files using the Gemini API, supporting metadata, system instructions, controlled output (JSON schema), and function calling.",
        formatter_class=argparse.RawTextHelpFormatter # Keep formatting in help
        )
    parser.add_argument("prompt_files",
                        metavar="PROMPT_FILE",
                        type=str,
                        nargs='+',
                        help="Path to one or more prompt files to process.\n\n"
                             "Prompt files can contain metadata lines (e.g., 'Model: model-name', 'Temperature: 0.5'),\n"
                             "and sections like '# System Instructions', '# Prompt', '# Controlled Output Schema', '# Functions'.\n\n"
                             "Supported Metadata:\n"
                             "  # RagEngine: <rag_corpus_resource_name> (e.g., projects/.../ragCorpora/123...)\n"
                             "  Model: <model_name> (e.g., gemini-1.5-flash-latest)\n"
                             "  Temperature: <float> (e.g., 0.7)\n"
                             "  Top P: <float> (e.g., 0.95)\n"
                             "  Top K: <int> (e.g., 40)\n"
                             "  Max Output Tokens: <int> (e.g., 1024)\n"
                             "  Stop Sequences: <comma-separated strings> (e.g., 'stop:,end:')\n"
                             "  Safety Settings: <comma-separated pairs> (e.g., 'harassment=none, hate_speech=block_medium_and_above')\n"
                             "    - Categories: harassment, hate_speech, sexually_explicit, dangerous_content\n"
                             "    - Thresholds: block_none (or none), block_low_and_above (or low),\n"
                             "                  block_medium_and_above (or medium), block_only_high (or high)\n\n"
                             "Sections:\n"
                             "  # System Instructions: Optional instructions for the model.\n"
                             "  # RagEngine: The display name of a Vertex AI Vector Search Index Endpoint to use for RAG.\n"
                             "  # RagEngine: The display name of a Vertex AI Vector Search Index Endpoint to use for RAG.\n"
                             "  # Prompt: The main user prompt.\n"
                             "  # Controlled Output Schema: Optional JSON schema for structured output.\n"
                             "    - Presence triggers JSON mode *if* '# Functions' is not present.\n"
                             "    - If JSON mode is active and the schema is successfully parsed, a second API call\n"
                             "      is made to generate a human-readable explanation of the JSON output.\n"
                             "  # Functions: Optional JSON list of function declarations for the model to call.\n"
                             "    - Presence enables function calling mode; overrides '# Controlled Output Schema' for response type.\n"
                             "    - Example Format:\n"
                             "    ```json\n"
                             "    [\n"
                             "      {\n"
                             "        \"name\": \"find_theaters\",\n"
                             "        \"description\": \"Find theaters showing movies near a location\",\n"
                             "        \"parameters\": {\n"
                             "          \"type\": \"object\",\n"
                             "          \"properties\": {\n"
                             "            \"location\": {\n"
                             "              \"type\": \"string\",\n"
                             "              \"description\": \"The city and state, e.g. San Francisco, CA\"\n"
                             "            },\n"
                             "            \"movie\": {\n"
                             "              \"type\": \"string\",\n"
                             "              \"description\": \"The name of the movie\"\n"
                             "            }\n"
                             "          },\n"
                             "          \"required\": [\"location\", \"movie\"]\n"
                             "        }\n"
                             "      }\n"
                             "    ]\n"
                             "    ```"
                        )

    args = parser.parse_args()

    # Use print for overall progress, logger for details/warnings/errors
    print(f"Starting processing for {len(args.prompt_files)} file(s)...")
    for prompt_file in args.prompt_files:
        call_gemini_with_prompt_file(prompt_file)
        print("-" * 30) # Separator between files

    print("Processing complete.")

if __name__ == "__main__":
    main()