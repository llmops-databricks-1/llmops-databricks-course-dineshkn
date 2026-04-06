# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.4: MLflow Log & Register Agent
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Running evaluation against the real agent
# MAGIC - Logging the agent model to MLflow with resource hints
# MAGIC - Registering the model to Unity Catalog
# MAGIC - Setting a model alias for serving
# MAGIC
# MAGIC ## ⚠️ Note: Run this notebook on Databricks (not locally)
# MAGIC
# MAGIC `mlflow.pyfunc.log_model(python_model=...)` with a file path needs to
# MAGIC resolve and upload the agent code to the MLflow artifact store, which
# MAGIC requires the Databricks runtime environment.

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

from causal_inference_curator.agent import log_register_agent
from causal_inference_curator.config import ProjectConfig
from causal_inference_curator.evaluation import (
    cites_sources_guideline,
    evaluate_agent,
    mentions_papers,
    polite_tone_guideline,
    word_count_check,
)

# COMMAND ----------

# Setup MLflow tracking (same as other notebooks)
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ.get("PROFILE", "llm-ops-course-dink")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load config directly via ProjectConfig (no Spark needed here)
cfg = ProjectConfig.from_yaml("../project_config.yml")
mlflow.set_experiment(cfg.experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Run Evaluation

# COMMAND ----------

eval_result = evaluate_agent(cfg, eval_inputs_path="../eval_inputs.txt")

logger.info("Evaluation complete. Metrics:")
for metric_name, value in eval_result.metrics.items():
    logger.info(f"  {metric_name}: {value:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Databricks Resources
# MAGIC
# MAGIC These resource hints tell Databricks Model Serving which endpoints and
# MAGIC indexes the agent needs access to at serve time.

# COMMAND ----------

resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksVectorSearchIndex(
        index_name=f"{cfg.catalog}.{cfg.schema}.causal_inference_chunks_index"
    ),
    DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build Model Config and Test Request

# COMMAND ----------

model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "system_prompt": cfg.system_prompt,
    "llm_endpoint": cfg.llm_endpoint,
}

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

test_request = {
    "input": [
        {
            "role": "user",
            "content": "What is the fundamental problem of causal inference?",
        }
    ],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log Model to MLflow

# COMMAND ----------

git_sha = "local"  # Replace with actual git SHA in CI/CD
run_id = "manual"

ts = datetime.now().strftime("%Y-%m-%d")

with mlflow.start_run(
    run_name=f"causal-inference-agent-{ts}",
    tags={"git_sha": git_sha, "run_id": run_id},
) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../causal_inference_agent.py",
        resources=resources,
        input_example=test_request,
        model_config=model_config,
    )
    mlflow.log_metrics(eval_result.metrics)

logger.info(f"Model logged: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register Model to Unity Catalog

# COMMAND ----------

model_name = f"{cfg.catalog}.{cfg.schema}.causal_inference_agent"

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags={"git_sha": git_sha, "run_id": run_id},
)

logger.info(f"Registered model: {model_name}")
logger.info(f"Version: {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Set Model Alias

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)

logger.info(f"Alias 'latest-model' → version {registered_model.version}")
logger.info("✓ Agent logged, evaluated, registered, and aliased!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. (Alternative) Use the Helper Function from agent.py
# MAGIC
# MAGIC The `log_register_agent` function in `causal_inference_curator.agent`
# MAGIC wraps steps 4–6 (log + register + alias) into a single call.
# MAGIC This is what you'd use in a CI/CD pipeline.

# COMMAND ----------

# This creates a new MLflow run and registers a new model version.
# It will increment the version number each time you run it.
registered_v2 = log_register_agent(
    cfg=cfg,
    git_sha="abc123",
    run_id="ci-run-456",
    agent_code_path="../causal_inference_agent.py",
    model_name=f"{cfg.catalog}.{cfg.schema}.causal_inference_agent",
    evaluation_metrics=eval_result.metrics,
)

logger.info(f"Registered via helper — version: {registered_v2.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Inspect Registered Model Versions

# COMMAND ----------

inspect_client = MlflowClient()
versions = inspect_client.search_model_versions(
    f"name='{cfg.catalog}.{cfg.schema}.causal_inference_agent'"
)

logger.info(f"Registered model: {cfg.catalog}.{cfg.schema}.causal_inference_agent")
logger.info("=" * 80)
for v in versions:
    aliases = v.aliases if hasattr(v, "aliases") else []
    logger.info(f"  Version {v.version} | status={v.status} | aliases={aliases} | run_id={v.run_id[:8]}...")
