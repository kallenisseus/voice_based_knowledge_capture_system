"""UI-facing helper functions.

This module intentionally stays thin so views can depend on one function
name even if the storage/query implementation changes later.
"""

from ai.repository import load_dashboard_data


def load_knowledge_base():
    """Return the complete payload required by the dashboard template."""
    return load_dashboard_data()
