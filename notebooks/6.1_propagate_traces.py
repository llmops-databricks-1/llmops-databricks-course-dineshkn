# Databricks notebook source
"""Generate inference traffic to populate MLflow traces for the Causal Inference Agent."""

import random
import time
from datetime import datetime

from databricks.sdk import WorkspaceClient
from openai import OpenAI

workspace = WorkspaceClient()
host = workspace.config.host
# Use config.authenticate() — works locally via VS Code metadata-service and on Databricks.
# workspace.tokens.create() is blocked in this workspace for API PAT creation.
token = workspace.config.authenticate()["Authorization"].split(" ")[1]

endpoint_name = "causal-inference-agent-endpoint-dev"

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

# COMMAND ----------
queries = [
    "What are the main methods for estimating causal effects from observational data?",
    "Explain the difference between average treatment effect and average treatment effect on the treated.",
    "What is the role of propensity scores in causal inference?",
    "How does instrumental variable estimation address endogeneity?",
    "What are the key assumptions behind difference-in-differences estimation?",
    "Summarize recent advances in regression discontinuity design methods.",
    "What is the potential outcomes framework and how is it used in causal research?",
    "How do researchers handle unmeasured confounders in observational studies?",
    "What are synthetic control methods and when are they most appropriate?",
    "Explain the concept of local average treatment effect and its limitations.",
    "What are the main identification strategies in applied microeconometrics?",
    "How is sensitivity analysis used to assess causal claims in observational studies?",
    "What are the recent developments in double machine learning for causal inference?",
    "How do researchers use natural experiments for causal identification?",
    "What is the LATE theorem and what does it tell us about IV estimates?",
    "Summarize common pitfalls in causal inference and how to avoid them.",
    "What are Directed Acyclic Graphs and how are they used in causal modeling?",
    "How does matching differ from regression for estimating treatment effects?",
    "What is the parallel trends assumption and how is it tested?",
    "Explain mediation analysis in the context of causal inference.",
    "What are heterogeneous treatment effects and how are they estimated?",
    "How is machine learning being integrated with causal inference methods?",
    "What is the role of randomization inference in testing causal hypotheses?",
    "How do researchers handle non-compliance in randomized controlled trials?",
    "What are the challenges of causal inference with high-dimensional data?",
    "Explain the concept of external validity in causal studies.",
    "What are recent developments in causal discovery algorithms?",
    "How is Bayesian reasoning applied to causal inference?",
    "What are the key differences between randomized and quasi-experimental designs?",
    "Summarize methodological advances in panel data methods for causal inference.",
]

# COMMAND ----------
for i, query in enumerate(queries):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
    request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

    print(f"[{i + 1}/{len(queries)}] {query[:60]}...")
    response = client.responses.create(
        model=endpoint_name,
        input=[{"role": "user", "content": query}],
        extra_body={
            "custom_inputs": {
                "session_id": session_id,
                "request_id": request_id,
            }
        },
    )
    time.sleep(2)

# COMMAND ----------
