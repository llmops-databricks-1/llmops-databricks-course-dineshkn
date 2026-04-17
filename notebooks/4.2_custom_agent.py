# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.2: CausalInferenceAgent with Tracing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Integrating MLflow tracing into the agent class
# MAGIC - Tracing LLM calls (streaming)
# MAGIC - Tracing tool executions (vector search)
# MAGIC - Session and request tracking
# MAGIC - End-to-end agent tracing
# MAGIC - Performance analysis

# COMMAND ----------

import os
import random
from datetime import datetime
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from causal_inference_curator.agent import CausalInferenceAgent
from causal_inference_curator.config import get_env, load_config

# COMMAND ----------

# Setup MLflow tracking
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ.get("PROFILE", "llm-ops-course-dink")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

mlflow.set_experiment(cfg.project.experiment_name)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Agent Architecture with Tracing
# MAGIC
# MAGIC ```
# MAGIC User Request
# MAGIC     ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │  @mlflow.trace(AGENT)                        │
# MAGIC │  predict_stream()                             │
# MAGIC │    ├─ @mlflow.trace(RETRIEVER)               │
# MAGIC │    │  load_memory(session_id)                 │
# MAGIC │    │  → prepend past messages                 │
# MAGIC │    │                                          │
# MAGIC │    ├─ @mlflow.trace(CHAIN)                   │
# MAGIC │    │  call_and_run_tools()                    │
# MAGIC │    │    ├─ mlflow.start_span(LLM)             │
# MAGIC │    │    │  call_llm()  ← streaming            │
# MAGIC │    │    │                                     │
# MAGIC │    │    ├─ @mlflow.trace(TOOL)               │
# MAGIC │    │    │  execute_tool()  ← vector search    │
# MAGIC │    │    │                                     │
# MAGIC │    │    └─ Loop until done                    │
# MAGIC │    │                                          │
# MAGIC │    └─ @mlflow.trace(CHAIN)                   │
# MAGIC │       save_memory(session_id, new_messages)   │
# MAGIC └──────────────────────────────────────────────┘
# MAGIC     ↓
# MAGIC Response + Complete Trace in MLflow UI
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the CausalInferenceAgent

# COMMAND ----------

agent = CausalInferenceAgent(
    llm_endpoint=cfg.project.llm_endpoint,
    system_prompt=cfg.project.system_prompt,
    catalog=cfg.project.catalog,
    schema=cfg.project.schema,
    # genie_space_id=cfg.project.genie_space_id,  # uncomment when Genie is configured
)

logger.info("✓ CausalInferenceAgent created with MCP tools:")
logger.info(f"  - Vector Search: {cfg.project.catalog}.{cfg.project.schema}")
logger.info(f"  - Tools: {list(agent._tools_dict.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Traced Agent — Single Turn

# COMMAND ----------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

test_request = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": "What is the fundamental problem of causal inference?",
        }
    ],
    custom_inputs={
        "session_id": session_id,
        "request_id": request_id,
    },
)

logger.info(f"Session ID: {session_id}")
logger.info(f"Request ID: {request_id}")
logger.info("Agent Response:")
logger.info("=" * 80)

response = agent.predict(test_request)
last_output = response.output[-1]
response_text = (
    last_output.content if hasattr(last_output, "content") else last_output["content"]
)
logger.info(response_text)

logger.info("✓ Trace created! Check MLflow UI for the complete trace.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Multi-Turn Conversation with Tracing

# COMMAND ----------

conversation_session = (
    f"s-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}"
)
logger.info(f"Conversation Session: {conversation_session}")

# Turn 1
request1 = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": "What is the difference-in-differences method?",
        }
    ],
    custom_inputs={
        "session_id": conversation_session,
        "request_id": f"req-1-{uuid4().hex[:8]}",
    },
)

response1 = agent.predict(request1)
last1 = response1.output[-1]
response1_text = last1.content if hasattr(last1, "content") else last1["content"]
logger.info("Turn 1:")
logger.info("User: What is the difference-in-differences method?")
logger.info(f"Agent: {response1_text}")

# Turn 2 — follow-up using conversation history
request2 = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "What is the difference-in-differences method?"},
        {"role": "assistant", "content": response1_text},
        {"role": "user", "content": "What are the key assumptions for this to be valid?"},
    ],
    custom_inputs={
        "session_id": conversation_session,
        "request_id": f"req-2-{uuid4().hex[:8]}",
    },
)

response2 = agent.predict(request2)
last2 = response2.output[-1]
response2_text = last2.content if hasattr(last2, "content") else last2["content"]
logger.info("Turn 2:")
logger.info("User: What are the key assumptions for this to be valid?")
logger.info(f"Agent: {response2_text}")

logger.info(f"✓ Multi-turn conversation traced with session: {conversation_session}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analysing Agent Traces

# COMMAND ----------

session_traces_df = mlflow.search_traces(
    filter_string=f"request_metadata.`mlflow.trace.session` = '{conversation_session}'",
    order_by=["timestamp_ms ASC"],
)

logger.info(f"Traces for session {conversation_session}:")
logger.info("=" * 80)

if len(session_traces_df) > 0:
    simple_cols = [
        c
        for c in session_traces_df.columns
        if c not in ["request", "response", "spans", "inputs", "outputs"]
    ]
    display(
        session_traces_df[simple_cols].head() if simple_cols else session_traces_df.head()
    )
else:
    logger.info("No traces found for this session.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Performance Analysis

# COMMAND ----------

recent_traces_df = mlflow.search_traces(
    order_by=["timestamp_ms DESC"],
    max_results=20,
)

if len(recent_traces_df) > 0:
    logger.info("Performance Statistics:")
    logger.info("=" * 80)
    logger.info(f"Total traces: {len(recent_traces_df)}")

    if "execution_time_ms" in recent_traces_df.columns:
        durations = recent_traces_df["execution_time_ms"].dropna()
        if len(durations) > 0:
            logger.info(f"Avg duration: {durations.mean():.2f}ms")
            logger.info(f"Min duration: {durations.min():.2f}ms")
            logger.info(f"Max duration: {durations.max():.2f}ms")

    if "status" in recent_traces_df.columns:
        logger.info("By Status:")
        for status, count in recent_traces_df["status"].value_counts().items():
            logger.info(f"  {status}: {count}")

    simple_cols = [
        c
        for c in recent_traces_df.columns
        if c not in ["request", "response", "spans", "inputs", "outputs"]
    ]
    display(
        recent_traces_df[simple_cols].head() if simple_cols else recent_traces_df.head()
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Trace Inspection

# COMMAND ----------

if len(recent_traces_df) > 0:
    trace = recent_traces_df.iloc[0]

    print("Detailed Trace Inspection:")
    print("=" * 80)
    print(f"Available columns: {list(recent_traces_df.columns)}")
    print()

    # Print all scalar fields
    for col in recent_traces_df.columns:
        val = trace.get(col)
        if col in ("tags", "request_metadata", "spans", "request", "response"):
            continue
        if val is not None:
            print(f"{col}: {val}")

    if trace.get("tags"):
        print("\nTags:")
        for k, v in trace["tags"].items():
            print(f"  {k}: {v}")

    if trace.get("request_metadata"):
        print("\nMetadata:")
        for k, v in trace["request_metadata"].items():
            print(f"  {k}: {v}")

    if "spans" in trace:
        spans_count = len(trace["spans"]) if trace["spans"] else 0
        print(f"\nSpans: {spans_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Tracing Best Practices Summary
# MAGIC
# MAGIC 1. **Always add `session_id`** for conversation tracking across turns
# MAGIC 2. **Use unique `request_id`** for each individual request
# MAGIC 3. **Include `git_sha`** so you can correlate traces to code versions
# MAGIC 4. **Trace all LLM calls** with `SpanType.LLM` for cost tracking
# MAGIC 5. **Trace all tool executions** with `SpanType.TOOL`
# MAGIC 6. **Use `SpanType.CHAIN`** for multi-step orchestration logic
# MAGIC 7. **Use `SpanType.RETRIEVER`** for memory / vector search loads
# MAGIC 8. **Set span inputs/outputs** — this is what shows up in the MLflow UI
# MAGIC 9. **Handle errors gracefully** — failed spans are still useful for debugging
# MAGIC 10. **Search traces** to analyse patterns across many requests
