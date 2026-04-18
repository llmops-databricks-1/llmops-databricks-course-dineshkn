# Databricks notebook source
# MAGIC %md
# MAGIC # 5.1b: Test Deployed Endpoint
# MAGIC
# MAGIC Quick test of the live `causal-inference-agent-endpoint-dev` endpoint.
# MAGIC Run locally with Databricks Connect — no runtime needed.

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv

    load_dotenv()
    profile = os.environ.get("PROFILE", "llm-ops-course-dink")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

endpoint_name = "causal-inference-agent-endpoint-dev"
workspace = WorkspaceClient()

host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Turn 1

# COMMAND ----------

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

logger.info(f"Session ID:  {session_id}")
logger.info(f"Response ID: {response.id}")
logger.info("-" * 80)
logger.info(response.output[0].content[0].text)
logger.info("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Turn 2 — same session (tests memory)

# COMMAND ----------

followup = client.responses.create(
    model=endpoint_name,
    input=[
        {
            "role": "user",
            "content": "Can you give a concrete example using potential outcomes notation?",
        }
    ],
    extra_body={
        "custom_inputs": {
            "session_id": session_id,
            "request_id": f"req-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}",
        }
    },
)

logger.info("Follow-up (same session — should have context):")
logger.info("-" * 80)
logger.info(followup.output[0].content[0].text)
logger.info("-" * 80)
