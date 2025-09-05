#!/bin/bash
# This script is executed by VS Code's integrated terminal on startup,
# as configured in .vscode/settings.json.

# First, source the user's standard .bashrc to get all their aliases and settings.
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Get the absolute path of the directory containing this script for robust pathing.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Then, source the project-specific configuration script.
echo "Sourcing project environment from .scripts/configure.sh..."
. .scripts/configure.sh