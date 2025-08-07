# Gemini Output for: configure.sh
## Request Configuration
- **Model:** gemini-1.5-flash-latest
- **System Instructions Provided:** No
- **'# Controlled Output Schema' Section Found:** No
- **'# Functions' Section Found:** No
- **Function Calling Active (Tools Provided):** No
- **JSON Output Mode Active (MIME Type):** False
- **Schema Parsed & Applied (for JSON Mode):** No
- **Safety Settings Applied:** [{'category': <HarmCategory.HARM_CATEGORY_HARASSMENT: 7>, 'threshold': <HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE: 2>}, {'category': <HarmCategory.HARM_CATEGORY_HATE_SPEECH: 8>, 'threshold': <HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE: 2>}, {'category': <HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: 9>, 'threshold': <HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE: 2>}, {'category': <HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 10>, 'threshold': <HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE: 2>}]
- **Timestamp:** 2025-08-07 18:16:15.193261

## Usage Metadata (Primary Call)
- **Prompt Token Count:** 965
- **Candidates Token Count:** 1291
- **Total Token Count:** 2256

## RAW OUTPUT

```text
This script sets up a Python virtual environment, installs dependencies, and installs a VS Code extension.  Here's an improved version with error handling, better logging, and more robust checks:

```bash
#!/bin/bash

# Usage: source .scripts/configure.sh

# --- Virtual Environment Setup ---
VENV_DIR=".venv/python3.12"
REQUIREMENTS_FILE=".scripts/requirements.txt"
CODE_OSS_EXEC="/opt/code-oss/bin/codeoss-cloudworkstations"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Python virtual environment '$VENV_DIR' not found."

  # Install python3-venv if needed
  if ! command -v python3-venv &> /dev/null; then
    echo "Installing python3-venv..."
    if ! sudo apt update && sudo apt install -y python3-venv; then
      echo "Error: Failed to install python3-venv." >&2
      exit 1
    fi
  fi

  # Create virtual environment
  echo "Creating Python virtual environment '$VENV_DIR'..."
  if ! /usr/bin/python3 -m venv "$VENV_DIR"; then
    echo "Error: Failed to create virtual environment." >&2
    exit 1
  fi

  # Install dependencies
  echo "Installing dependencies into '$VENV_DIR' from '$REQUIREMENTS_FILE'..."
  if ! "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"; then
    echo "Error: Failed to install dependencies." >&2
    exit 1
  fi
fi

# --- Install required utilities ---
install_util() {
  local util="$1"
  if ! command -v "$util" &> /dev/null; then
    echo "'$util' command not found. Attempting to install..."
    if ! sudo apt-get update && sudo apt-get install -y "$util"; then
      echo "Error: Failed to install '$util'." >&2
      exit 1
    fi
  fi
}

install_util unzip
install_util jq


# --- VS Code Extension Setup (One-time) ---
VSIX_URL="https://marketplace.visualstudio.com/_apis/public/gallery/publishers/emeraldwalk/vsextensions/RunOnSave/0.3.2/vspackage" #Updated URL - consider fetching latest
VSIX_FILE="/tmp/emeraldwalk.runonsave.vsix"

check_vscode_extension() {
    if ! $CODE_OSS_EXEC --list-extensions | grep -q "emeraldwalk.runonsave"; then
        return 0
    else
        return 1
    fi
}

if check_vscode_extension; then
    echo "Checking for 'emeraldwalk.runonsave' VS Code extension..."
    echo "Downloading extension from '$VSIX_URL'..."

    if curl --fail -L -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" -o "$VSIX_FILE" "$VSIX_URL"; then
        if unzip -t "$VSIX_FILE" &> /dev/null; then
            echo "Download complete. Installing..."
            if $CODE_OSS_EXEC --install-extension "$VSIX_FILE"; then
                echo "Extension 'emeraldwalk.runonsave' installed successfully."
                echo "IMPORTANT: Please reload the VS Code window to activate the extension."
            else
                echo "Error: Failed to install the extension from '$VSIX_FILE'." >&2
                exit 1
            fi
        else
            echo "Error: Downloaded file is not a valid VSIX package." >&2
            exit 1
        fi
    else
        echo "Error: Failed to download extension from '$VSIX_URL'." >&2
        exit 1
    fi
else
  echo "'emeraldwalk.runonsave' already installed."
fi


echo "Configuration complete."
```

**Improvements:**

* **Error Handling:**  Uses `if ... then ... else ... fi` and `exit 1` to handle errors gracefully.  Error messages are sent to standard error (`>&2`).
* **Variable Usage:** Uses variables for paths and URLs, making the script easier to maintain and read.
* **Function for Utility Installation:** The `install_util` function avoids code duplication.
* **Updated VSIX URL:** The VSIX URL now directly points to the VSIX package. While using a static URL is less ideal than dynamically determining the latest version, I've adjusted it to a more direct and accurate location.  Consider implementing a mechanism to get the latest version if possible (e.g., using the VS Code Marketplace API).
* **More informative logging:** Provides clearer messages about each step.
* **Modern curl user agent:** Uses a more up-to-date user agent string for the curl request.
* **Check for existing extension:** Added `check_vscode_extension` to avoid unnecessary download and installation if the extension is already present.
* **Shebang:** Added `#!/bin/bash` at the beginning to explicitly specify the interpreter.  Make this script executable with `chmod +x .scripts/configure.sh`.

Remember to replace `/opt/code-oss/bin/codeoss-cloudworkstations` with the correct path to your VS Code executable if it's different.  Also, consider adding more robust error handling around network issues during the `curl` command.  A mechanism to fetch the latest version of the extension from the VS Code Marketplace would be a substantial improvement over relying on a hardcoded version.

```
