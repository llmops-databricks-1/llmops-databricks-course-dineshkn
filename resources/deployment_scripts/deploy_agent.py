# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
# MAGIC
# MAGIC Deploys the registered causal inference agent to a Databricks Model Serving endpoint.
# MAGIC Passed parameters via `base_parameters` in the DAB job YAML:
# MAGIC   - `env`: target environment (dev / acc / prd)
# MAGIC   - `git_sha`: git commit SHA for traceability
# MAGIC
# MAGIC ⚠️ Must run on Databricks — uses `dbutils.secrets` and `agents.deploy()`.

# COMMAND ----------

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient

from causal_inference_curator.config import ProjectConfig
from causal_inference_curator.utils.common import get_widget

# COMMAND ----------

env = get_widget("env", "dev")
git_sha = get_widget("git_sha", "local")

# Secret scope follows the course convention: dev_SPN, acc_SPN, prd_SPN
secret_scope = f"{env}_SPN"

cfg = ProjectConfig.from_yaml("../../project_config.yml", env=env)

model_name = f"{cfg.catalog}.{cfg.schema}.causal_inference_agent"
endpoint_name = f"causal-inference-agent-endpoint-{env}"

client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "latest-model").version

# set_experiment creates it if it doesn't exist, always returns a valid object
experiment = mlflow.set_experiment(cfg.experiment_name)

logger.info("Deploying agent:")
logger.info(f"  Model:      {model_name}")
logger.info(f"  Version:    {model_version}")
logger.info(f"  Endpoint:   {endpoint_name}")
logger.info(f"  Scope:      {secret_scope}")
logger.info(f"  Experiment: {experiment.experiment_id}")

# COMMAND ----------

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
        "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client_id}}}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client_secret}}}}",
        "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
    },
)

logger.info("✓ Deployment triggered — endpoint will be ready in 5-10 minutes.")
