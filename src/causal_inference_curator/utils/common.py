"""Shared utilities for notebooks and deployment scripts."""

import os

import mlflow
from databricks.sdk.runtime import dbutils
from dotenv import load_dotenv


def get_widget(name: str, default: str | None = None) -> str | None:
    """Get a Databricks widget value with a fallback default.

    Args:
        name: Widget name.
        default: Default value if widget is not set.

    Returns:
        Widget value or default.
    """
    try:
        return dbutils.widgets.get(name)
    except KeyError:
        return default


def set_mlflow_tracking_uri() -> None:
    """Set MLflow tracking and registry URIs for local Databricks Connect execution."""
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        load_dotenv()
        profile = os.environ.get("PROFILE", "llm-ops-course-dink")
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
