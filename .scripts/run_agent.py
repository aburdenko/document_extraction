#!/usr/bin/env python3
# Run with: ./.scripts/run_agent.py prompts/sample-agent-task.md

import os
import sys
import argparse
import uuid
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool, Part, FunctionDeclaration
from google.cloud import logging as cloud_logging
from google.cloud.logging.handlers import setup_logging

# Import our defined tools
from agent_tools import get_todays_date

# --- Constants and Config ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 10 # Prevent infinite loops

# Regex to find sections like # Name, # Instruction, etc.
SECTION_PATTERN = re.compile(r"^\s*#+\s+([\w\s]+)\s*$", re.MULTILINE)

# --- Tool and Logging Setup ---
AVAILABLE_TOOLS = {
    # This dictionary maps function tool names from markdown to actual Python functions.
    "get_todays_date": get_todays_date,
}

def setup_cloud_logger():
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        logger.warning("PROJECT_ID not set. Skipping Cloud Logging setup.")
        return None, None
    try:
        log_name = os.getenv("LOG_NAME", "run_gemini_from_file")
        client = cloud_logging.Client(project=project_id)
        handler = cloud_logging.handlers.CloudLoggingHandler(client, name=log_name)
        setup_logging(handler)
        logger.info(f"Cloud Logging setup for log name: '{log_name}'.")
        return client, handler
    except Exception as e:
        logger.warning(f"Could not set up Cloud Logging: {e}.")
        return None, None

def log_to_cloud(payload: Dict[str, Any]):
    """Logs a structured payload to Google Cloud Logging."""
    logger.info("Agent Log", extra={'json_fields': payload})

class AgentLogger:
    """A helper class to manage consistent logging for an agent session."""
    def __init__(self, session_id: str, instruction: str, initial_prompt: str, ground_truth: str):
        self.base_payload = {
            "session_id": session_id,
            "instruction": instruction,
            "initial_prompt": initial_prompt,
            "ground_truth": ground_truth,
        }

    def log(self, payload: Dict[str, Any]):
        """Merges the given payload with the base session payload and logs it."""
        full_payload = self.base_payload.copy()
        full_payload.update(payload)
        log_to_cloud(full_payload)

def parse_sections(text_content: str) -> Dict[str, str]:
    """
    Parses the text content into sections based on headings and strips
    markdown code fences from each section's content.
    """
    sections: Dict[str, str] = {}
    last_pos = 0
    # Content before the first heading is captured as 'initial_content'.
    current_section_name = "initial_content"

    for match in SECTION_PATTERN.finditer(text_content):
        if current_section_name is not None:
            start, _ = match.span()
            section_content = text_content[last_pos:start].strip()
            sections[current_section_name] = section_content

        # Convert heading to key: lowercase, replace spaces with underscores
        current_section_name = match.group(1).strip().lower().replace(" ", "_")
        _, last_pos = match.span()

    if current_section_name is not None:
        sections[current_section_name] = text_content[last_pos:].strip()

    # After parsing, strip code fences from all section values for robustness.
    for key, value in sections.items():
        # Use \w* to match any language identifier (json, plaintext, etc.) or none.
        stripped_value = re.sub(r"^\s*```\w*\s*", "", value, flags=re.IGNORECASE | re.MULTILINE)
        stripped_value = re.sub(r"\s*```\s*$", "", stripped_value, flags=re.MULTILINE).strip()
        sections[key] = stripped_value

    return sections

# --- Core Agent Logic ---
def run_agent(prompt_filepath: str):
    filepath = Path(prompt_filepath)
    session_id = f"agent-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    print(f"--- Starting Agent Session: {session_id} for file: {filepath.name} ---")

    # 1. Parse the task file
    file_content = filepath.read_text()
    sections = parse_sections(file_content)

    agent_name = sections.get("name", session_id)
    # Prioritize 'instruction', fall back to 'system_instructions', then to a default.
    system_instruction = sections.get("instruction") or sections.get("system_instructions") or "You are a helpful agent."
    initial_prompt = sections.get("prompt", "")
    ground_truth = sections.get("ground_truth", "")
    tools_json_str = sections.get("tools", "[]")

    # Fallback logic: If no '# Prompt' section, use the initial content.
    if not initial_prompt and 'initial_content' in sections:
        initial_prompt = sections.get('initial_content', '').strip()
        if initial_prompt:
            logger.warning("No '# Prompt' section found. Using content before the first heading as the initial prompt.")

    if not initial_prompt:
        logger.error("The '# Prompt' section is missing or empty in the task file, and no fallback content was found. The agent cannot start without an initial prompt.")
        # Exit gracefully without creating an output file, as no work was done.
        return

    # Instantiate the logger with the base information for the session.
    agent_logger = AgentLogger(
        session_id=session_id,
        instruction=system_instruction,
        initial_prompt=initial_prompt,
        ground_truth=ground_truth
    )

    # Code fence stripping is now handled robustly in the parse_sections function.
    # 2. Setup Model and Tools
    model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
    model = GenerativeModel(
        model_name,
        system_instruction=[system_instruction]
    )
    
    all_tools = []
    function_declarations = []
    # Only attempt to parse JSON if the tools string is not empty.
    if tools_json_str:
        try:
            tools_list = json.loads(tools_json_str)
            for tool_config in tools_list:
                tool_type = tool_config.get("type")
                if tool_type == "VertexAiRagRetrieval":
                    rag_corpus = tool_config.get("rag_corpus")
                    if rag_corpus:
                        # The RagRetrievalConfig expects 'top_k', not 'similarity_top_k'.
                        # We check for both for backward compatibility with the markdown file.
                        retrieval_config = rag.RagRetrievalConfig(
                            top_k=tool_config.get("top_k", tool_config.get("similarity_top_k", 10)),
                            # The 'distance_threshold' parameter is no longer supported in this class in the latest SDK.
                        )
                        # The retrieval_config should be passed to the VertexRagStore source, not to the Retrieval object.
                        retrieval = rag.Retrieval(
                            source=rag.VertexRagStore(
                                rag_resources=[rag.RagResource(rag_corpus=rag_corpus)], rag_retrieval_config=retrieval_config
                            )
                        )
                        rag_tool = Tool.from_retrieval(retrieval)
                        all_tools.append(rag_tool)
                        logger.info(f"Configured RAG tool '{tool_config.get('name')}' with corpus: {rag_corpus}")
                elif tool_type == "FunctionTool":
                    tool_name = tool_config.get("name")
                    if tool_name in AVAILABLE_TOOLS:
                        func = AVAILABLE_TOOLS[tool_name]
                        function_declarations.append(FunctionDeclaration.from_func(func))
                        logger.info(f"Found Function tool: '{tool_name}'")
                    else:
                        logger.warning(f"Function tool '{tool_name}' defined in markdown but not found in agent_tools.py")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON in # Tools section: {e}")

    if function_declarations:
        all_tools.append(Tool(function_declarations=function_declarations))
        logger.info(f"Configured {len(function_declarations)} function-based tools.")

    # 3. Run the Agent Loop
    conversation_history = [initial_prompt]
    final_answer = ""

    for step in range(MAX_AGENT_STEPS):
        print(f"\n[Step {step + 1}] Thinking...")
        # Pass the dynamically built list of tools to the model
        response = model.generate_content(conversation_history, tools=all_tools)
        part = response.candidates[0].content.parts[0]

        # New: Check for and display grounding metadata from the RAG tool
        if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata.retrieval_queries:
            print("\n--- Grounding Metadata (Retrieved from RAG Engine) ---")
            for query in response.candidates[0].grounding_metadata.retrieval_queries:
                for chunk in query.retrieved_chunks:
                     print(f"  - Source: {chunk.source}")
                     print(f"  - Content: {chunk.content[:150]}...")
            print("----------------------------------------------------")

        if part.function_call:
            function_call = part.function_call
            tool_name = function_call.name
            tool_args = {key: value for key, value in function_call.args.items()}
            print(f"Action: Calling tool '{tool_name}' with args: {tool_args}")

            agent_logger.log({
                "step": step, "log_type": "thought",
                "thought": f"Decided to call tool '{tool_name}' with arguments {tool_args}."
            })

            if tool_name in AVAILABLE_TOOLS:
                tool_function = AVAILABLE_TOOLS[tool_name]
                try:
                    tool_output = tool_function(**tool_args)
                    print(f"Observation: {tool_output}")
                    agent_logger.log({
                        "step": step, "log_type": "tool_result",
                        "tool_name": tool_name, "tool_args": tool_args, "tool_output": tool_output
                    })
                    conversation_history.append(response.candidates[0].content) # Add model's function call
                    conversation_history.append(Part.from_function_response(name=tool_name, response={"result": tool_output}))
                except Exception as e:
                    print(f"Error executing tool '{tool_name}': {e}")
                    # Feed the error back to the model
                    conversation_history.append(Part.from_function_response(name=tool_name, response={"error": str(e)}))
            else:
                print(f"Error: Agent tried to call unknown tool '{tool_name}'")
                conversation_history.append(Part.from_function_response(name=tool_name, response={"error": f"Tool '{tool_name}' not found."}))
        else:
            # Safely get the text attribute. If it's missing, the model has finished
            # without a clear text response, which can happen. Default to an empty string.
            final_answer = getattr(part, 'text', '')
            if not final_answer:
                logger.warning("Model finished without a function call or a text response. The agent might be stuck or has completed its task implicitly.")
                final_answer = "(No final text answer provided by the model)"
            print(f"\nFinal Answer: {final_answer}")
            agent_logger.log({
                "step": step, "log_type": "final_answer", "final_answer": final_answer
            })
            break # Agent has finished

    # 4. Save Output
    output_filename = filepath.with_name(f"{filepath.stem}.output.md")
    output_content = f"# Agent Output for: {filepath.name}\n\n"
    output_content += f"**Instruction:** {system_instruction}\n\n"
    output_content += f"**Final Answer:**\n{final_answer}\n"
    output_filename.write_text(output_content)
    print(f"\n--- Agent Session Finished. Output saved to: {output_filename} ---")


def main():
    parser = argparse.ArgumentParser(description="Run an agent based on a task defined in a markdown file.")
    parser.add_argument("prompt_file", type=str, help="Path to the agent task file.")
    args = parser.parse_args()

    logging_client, cloud_logging_handler = setup_cloud_logger()
    
    try:
        run_agent(args.prompt_file)
    finally:
        # Ensure logs are sent before the script exits.
        if cloud_logging_handler and hasattr(cloud_logging_handler, 'transport'):
            try:
                logger.info("Flushing Cloud Logging handler to send pending logs...")
                cloud_logging_handler.transport.flush()
            except Exception as e:
                # Use print for final errors as logging might be shutting down.
                print(f"Error flushing Cloud Logging handler: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()