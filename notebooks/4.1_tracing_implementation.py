# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.1: MLflow Tracing Implementation
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is tracing?
# MAGIC - Why tracing matters for GenAI
# MAGIC - Using @mlflow.trace decorator
# MAGIC - Manual span creation
# MAGIC - Adding metadata and tags
# MAGIC - Searching and analyzing traces

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from mlflow.entities import SpanType
from openai import OpenAI
from pyspark.sql import SparkSession

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Tracing?
# MAGIC
# MAGIC **Tracing** captures the execution flow of your GenAI application.
# MAGIC
# MAGIC ### Why Tracing Matters:
# MAGIC
# MAGIC - **Observability**: See what your agent is doing step by step
# MAGIC - **Debugging**: Find exactly where things go wrong
# MAGIC - **Performance**: Identify bottlenecks (slow retrieval? slow LLM?)
# MAGIC - **Cost**: Track token usage per request
# MAGIC - **Quality**: Analyse outputs over time
# MAGIC
# MAGIC ### Trace Structure for our Causal Inference Agent:
# MAGIC
# MAGIC ```
# MAGIC Trace (Root — AGENT)
# MAGIC ├── Span: memory_load (RETRIEVER)  ← load past conversation from Lakebase
# MAGIC ├── Span: call_and_run_tools (CHAIN)
# MAGIC │   ├── Span: call_llm (LLM)       ← first LLM call
# MAGIC │   ├── Span: execute_tool (TOOL)  ← vector search for papers
# MAGIC │   └── Span: call_llm (LLM)       ← LLM call with search results
# MAGIC └── Span: memory_save (CHAIN)      ← persist new messages
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simple Tracing with @mlflow.trace

# COMMAND ----------

mlflow.set_experiment(cfg.project.experiment_name)


@mlflow.trace
def add_numbers(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


result = add_numbers(5, 3)
logger.info(f"Result: {result}")
logger.info("✓ Trace created! Check MLflow UI to see the trace.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tracing with Span Types

# COMMAND ----------


@mlflow.trace(span_type=SpanType.LLM)
def call_llm_stub(prompt: str) -> str:
    """Simulate an LLM call."""
    return f"Response to: {prompt}"


@mlflow.trace(span_type=SpanType.TOOL)
def search_papers_stub(query: str) -> list:
    """Simulate a vector search for causal inference papers."""
    return [
        {
            "id": 1,
            "title": "Identification of Causal Effects Using Instrumental Variables",
        },
        {"id": 2, "title": "Estimating Average Treatment Effects"},
    ]


@mlflow.trace(span_type=SpanType.CHAIN)
def process_query(user_query: str) -> str:
    """Process a user query: search papers then call LLM with results."""
    results = search_papers_stub(user_query)
    prompt = f"User asked: {user_query}\nRelevant papers: {results}"
    return call_llm_stub(prompt)


output = process_query("What papers discuss instrumental variables?")
logger.info(f"Output: {output}")
logger.info("✓ Multi-span trace created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Manual Span Creation

# COMMAND ----------


def score_relevance(query: str, results: list) -> dict:
    """Score and rank search results — demonstrates manual span nesting."""

    with mlflow.start_span("score_relevance") as span:
        span.set_inputs({"query": query, "num_results": len(results)})

        with mlflow.start_span("compute_scores") as step1:
            scores = [0.95, 0.87, 0.72][: len(results)]
            step1.set_outputs({"scores": scores})

        with mlflow.start_span("rank_results") as step2:
            ranked = sorted(
                zip(results, scores, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
            top_result = ranked[0][0] if ranked else None
            step2.set_outputs({"top_result": top_result})

        span.set_outputs({"top_result": top_result, "scores": scores})
        return {"top_result": top_result, "scores": scores}


result = score_relevance(
    "instrumental variables",
    ["Paper A", "Paper B", "Paper C"],
)
logger.info(f"Top result: {result['top_result']}")
logger.info("✓ Trace with nested spans created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Adding Metadata and Tags

# COMMAND ----------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"
git_sha = "abc123def456"


@mlflow.trace
def function_with_metadata(x: int, y: int) -> int:
    """Demonstrate rich trace metadata."""

    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
            "user_id": "researcher_001",
            "environment": "dev",
        },
        tags={
            "model_serving_endpoint_name": "causal-inference-agent-endpoint",
            "model_version": "1",
            "git_sha": git_sha,
            "request_type": "paper_search",
        },
        client_request_id=request_id,
    )

    return x + y


result = function_with_metadata(10, 20)
logger.info(f"Result: {result}")
logger.info(f"Session ID: {session_id}")
logger.info(f"Request ID: {request_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Searching Traces

# COMMAND ----------

traces_df = mlflow.search_traces(
    filter_string=f"tags.git_sha = '{git_sha}'",
    max_results=5,
)

logger.info(f"Found {len(traces_df)} traces with git_sha={git_sha}")

if len(traces_df) > 0:
    cols_to_show = [
        c
        for c in ["request_id", "timestamp_ms", "status", "tags"]
        if c in traces_df.columns
    ]
    display(traces_df[cols_to_show].head() if cols_to_show else traces_df.head())
else:
    logger.info("No traces found. Run some traced functions first!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tracing Real LLM Calls

# COMMAND ----------

w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints",
)


@mlflow.trace(span_type=SpanType.LLM)
def call_real_llm(prompt: str, model: str | None = None) -> str:
    """Call a real Databricks-hosted LLM with tracing."""
    model = model or cfg.project.llm_endpoint

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a causal inference expert."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.7,
    )

    return response.choices[0].message.content


result = call_real_llm(
    "What is the fundamental problem of causal inference in one sentence?"
)
logger.info(f"LLM Response: {result}")
logger.info("✓ Real LLM call traced!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Tracing a Full Agent Interaction (Simulated)

# COMMAND ----------


@mlflow.trace(span_type=SpanType.AGENT)
def agent_interaction(user_message: str) -> dict:
    """Simulate a complete causal inference agent interaction."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
    request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

    mlflow.update_current_trace(
        metadata={"mlflow.trace.session": session_id},
        tags={"agent_type": "causal_inference_research", "model_version": "1.0"},
        client_request_id=request_id,
    )

    with mlflow.start_span("analyze_query", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"query": user_message})
        analysis = {"intent": "paper_search", "topic": "causal inference"}
        span.set_outputs(analysis)

    with mlflow.start_span("search_papers", span_type=SpanType.TOOL) as span:
        span.set_inputs({"query": analysis["topic"]})
        results = [
            {"title": "Identification of Causal Effects", "relevance": 0.95},
            {"title": "Potential Outcomes Framework", "relevance": 0.91},
        ]
        span.set_outputs({"results": results})

    with mlflow.start_span("generate_response", span_type=SpanType.LLM) as span:
        span.set_inputs({"user_message": user_message, "search_results": results})
        response = (
            f"I found {len(results)} relevant papers on {analysis['topic']}. "
            f"The most relevant is '{results[0]['title']}'."
        )
        span.set_outputs({"response": response})

    return {"response": response, "session_id": session_id, "request_id": request_id}


result = agent_interaction("What papers discuss the potential outcomes framework?")
logger.info(f"Agent Response: {result['response']}")
logger.info(f"Session ID: {result['session_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Analysing Recent Traces

# COMMAND ----------

recent_traces_df = mlflow.search_traces(
    order_by=["timestamp_ms DESC"],
    max_results=10,
)

logger.info(f"Recent Traces: {len(recent_traces_df)}")

if len(recent_traces_df) > 0:
    simple_cols = [
        c
        for c in recent_traces_df.columns
        if c not in ["request", "response", "spans", "inputs", "outputs"]
    ]
    display(
        recent_traces_df[simple_cols].head(10)
        if simple_cols
        else recent_traces_df.head(10)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Trace Filtering Examples

# COMMAND ----------

failed_traces = mlflow.search_traces(filter_string="status = 'ERROR'", max_results=5)
logger.info(f"Failed traces: {len(failed_traces)}")

endpoint_traces = mlflow.search_traces(
    filter_string="tags.model_serving_endpoint_name = 'causal-inference-agent-endpoint'",
    max_results=5,
)
logger.info(f"Traces for causal inference endpoint: {len(endpoint_traces)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Best Practices
# MAGIC
# MAGIC ### Do:
# MAGIC 1. Use appropriate span types (LLM, TOOL, CHAIN, AGENT, RETRIEVER)
# MAGIC 2. Add `session_id` for conversation tracking
# MAGIC 3. Add `request_id` for individual request tracking
# MAGIC 4. Include `git_sha` for version tracking
# MAGIC 5. Set inputs and outputs on every span
# MAGIC 6. Trace all LLM calls for cost tracking
# MAGIC 7. Use nested spans for complex operations
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Store sensitive data (tokens, credentials) in traces
# MAGIC 2. Trace too granularly — performance overhead adds up
# MAGIC 3. Forget to add metadata — it makes debugging much harder
# MAGIC 4. Skip tracing expensive operations (vector search, LLM calls)
