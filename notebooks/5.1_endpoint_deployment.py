# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 5.1: Agent Deployment & Testing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Deploying agents using `agents.deploy()`
# MAGIC - Configuring environment variables and secrets
# MAGIC - Testing deployed endpoints
# MAGIC - Using OpenAI-compatible client
# MAGIC
# MAGIC ## ⚠️ Note: Run this notebook on Databricks (not locally)
# MAGIC
# MAGIC `agents.deploy()` and `dbutils.secrets` require the Databricks runtime.
# MAGIC The endpoint testing section (Step 3) can be run locally if preferred.

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient

from causal_inference_curator.config import ProjectConfig

# Setup MLflow tracking (no-op when running on Databricks runtime)
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv

    load_dotenv()
    profile = os.environ.get("PROFILE", "llm-ops-course-dink")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

cfg = ProjectConfig.from_yaml("../project_config.yml")

model_name = f"{cfg.catalog}.{cfg.schema}.causal_inference_agent"
endpoint_name = "causal-inference-agent-endpoint-dev"
secret_scope = "dev_SPN"

model_version = (
    MlflowClient().get_model_version_by_alias(model_name, "latest-model").version
)

workspace = WorkspaceClient()
experiment = MlflowClient().get_experiment_by_name(cfg.experiment_name)

logger.info(f"Model:    {model_name} (version {model_version})")
logger.info(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Deploy Agent
# MAGIC
# MAGIC The `agents.deploy()` API handles:
# MAGIC - Endpoint creation and configuration
# MAGIC - Inference tables for monitoring
# MAGIC - Environment variables and secrets (referenced via `{{secrets/scope/key}}` syntax)
# MAGIC - Model versioning
# MAGIC
# MAGIC The `dev_SPN` secret scope is pre-configured in the course workspace with
# MAGIC `client_id` and `client_secret` keys for the service principal used at serve time.

# COMMAND ----------

git_sha = "local"

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "LAKEBASE_SP_CLIENT_ID": f"{{secrets/{secret_scope}/client_id}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{secrets/{secret_scope}/client_secret}}",
        "LAKEBASE_SP_HOST": workspace.config.host,
    },
)

logger.info("✓ Deployment triggered — endpoint will be ready in 5-10 minutes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Wait for Deployment
# MAGIC
# MAGIC After triggering deployment, wait 5–10 minutes for the endpoint to become ready.
# MAGIC You can monitor progress in the Databricks UI under **Serving → Endpoints**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Deployed Endpoint
# MAGIC
# MAGIC Use the OpenAI-compatible Responses API to query the live endpoint.
# MAGIC Generate a short-lived personal token via the Workspace SDK.

# COMMAND ----------

host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

from openai import OpenAI

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = client.responses.create(
    model=endpoint_name,
    input=[
        {
            "role": "user",
            "content": "What is the fundamental problem of causal inference?",
        }
    ],
    extra_body={
        "custom_inputs": {
            "session_id": session_id,
            "request_id": request_id,
        }
    },
)

logger.info(f"Response ID: {response.id}")
logger.info(f"Session ID:  {response.custom_outputs.get('session_id')}")
logger.info(f"Request ID:  {response.custom_outputs.get('request_id')}")
logger.info("\nAssistant Response:")
logger.info("-" * 80)
logger.info(response.output[0].content[0].text)
logger.info("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test Multi-Turn Conversation
# MAGIC
# MAGIC Re-use the same `session_id` to verify Lakebase memory carries context across turns.

# COMMAND ----------

followup_request_id = (
    f"req-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}"
)

followup = client.responses.create(
    model=endpoint_name,
    input=[
        {
            "role": "user",
            "content": "Can you give me a concrete example with potential outcomes notation?",
        }
    ],
    extra_body={
        "custom_inputs": {
            "session_id": session_id,
            "request_id": followup_request_id,
        }
    },
)

logger.info("Follow-up response (same session — should have context from previous turn):")
logger.info("-" * 80)
logger.info(followup.output[0].content[0].text)
logger.info("-" * 80)
