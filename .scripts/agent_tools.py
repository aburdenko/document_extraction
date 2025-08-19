"""
This file defines the tools available to our agent.
Each function represents a tool the agent can decide to use.
"""

from datetime import datetime

def get_todays_date() -> str:
    """Returns the current date as a string."""
    print("TOOL: Getting today's date...")
    return datetime.now().strftime("%Y-%m-%d")