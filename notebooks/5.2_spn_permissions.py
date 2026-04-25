# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 5.2: SPN Permissions
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Granting a Service Principal (SPN) the permissions it needs at serve time
# MAGIC - Vector Search endpoint access
# MAGIC - SQL Warehouse access
# MAGIC
# MAGIC ## ⚠️ Note: Run this notebook on Databricks (not locally)
# MAGIC
# MAGIC Uses `dbutils.secrets` which is only available on the Databricks runtime.
# MAGIC
# MAGIC ## Note on Genie
# MAGIC
# MAGIC The reference notebook also grants Genie permissions, but our `genie_space_id`
# MAGIC is `null` so that block is omitted here.

# COMMAND ----------

# MAGIC %pip install loguru
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
from loguru import logger

# Hardcoded — this notebook has no wheel dependency
vector_search_endpoint = "llmops_course_vs_endpoint"

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Get SPN Application ID
# MAGIC
# MAGIC The `dev_SPN` secret scope is pre-configured in the course workspace.
# MAGIC Print trick below lets you verify the value without fully exposing it.

# COMMAND ----------

spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")

# Safe print — shows all but last char, then last char with a space separator
logger.info(
    "SPN app_id (masked): "
    + dbutils.secrets.get("dev_SPN", "client_id")[:-1]
    + " "
    + dbutils.secrets.get("dev_SPN", "client_id")[-1]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Grant Vector Search Endpoint Access

# COMMAND ----------

vs_endpoint = w.vector_search_endpoints.get_endpoint(vector_search_endpoint)
w.permissions.update(
    request_object_type="vector-search-endpoints",
    request_object_id=vs_endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)

logger.info(f"✓ Granted CAN_USE on vector search endpoint: {vector_search_endpoint}")

# COMMAND ----------

logger.info("✓ SPN permissions configured.")
