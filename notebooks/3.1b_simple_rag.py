# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1b: Simple RAG with Vector Search + Session Memory
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Retrieval-Augmented Generation (RAG)
# MAGIC - Vector search for document retrieval
# MAGIC - Session memory with Lakebase
# MAGIC - Multi-turn stateful conversations
# MAGIC
# MAGIC **RAG Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Load session history (Lakebase)
# MAGIC     ↓
# MAGIC Vector Search (retrieve relevant papers)
# MAGIC     ↓
# MAGIC Build Prompt (question + context + history)
# MAGIC     ↓
# MAGIC LLM → Response
# MAGIC     ↓
# MAGIC Save updated history (Lakebase)
# MAGIC ```

# COMMAND ----------

from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config
from causal_inference_curator.memory import LakebaseMemory

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

client = OpenAI(
    api_key=w.config.authenticate()["Authorization"].split(" ")[1],
    base_url=f"{w.config.host}/serving-endpoints",
)

vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.config.authenticate()["Authorization"].split(" ")[1],
    disable_notice=True,
)

logger.info(f"✓ Connected to workspace: {w.config.host}")
logger.info(f"✓ Using LLM endpoint: {cfg.project.llm_endpoint}")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create or Get Lakebase Instance
# MAGIC
# MAGIC **Lakebase** is Databricks' managed PostgreSQL.
# MAGIC Run this cell once to provision the instance; subsequent runs will reuse it.

# COMMAND ----------

INSTANCE_NAME = "causal-inference-agent-instance"

# Get usage_policy_id from your Databricks admin if required.
# On many workspaces this field is optional — try without it first.
USAGE_POLICY_ID = None  # Set to your policy ID if instance creation fails

try:
    instance = w.database.get_database_instance(INSTANCE_NAME)
    logger.info(f"✓ Using existing Lakebase instance: {INSTANCE_NAME}")
    if instance.state == DatabaseInstanceState.STOPPED:
        logger.info("Instance stopped — restarting...")
        w.database.update_database_instance(
            name=INSTANCE_NAME,
            database_instance=DatabaseInstance(name=INSTANCE_NAME, stopped=False),
            update_mask="stopped",
        )
        instance = w.database.wait_get_database_instance_database_available(INSTANCE_NAME)
except Exception:
    logger.info(f"Creating new Lakebase instance: {INSTANCE_NAME}")
    db_instance = DatabaseInstance(name=INSTANCE_NAME, capacity="CU_1")
    if USAGE_POLICY_ID:
        db_instance.usage_policy_id = USAGE_POLICY_ID
    instance = w.database.create_database_instance(db_instance)
    instance = w.database.wait_get_database_instance_database_available(INSTANCE_NAME)

lakebase_host = instance.read_write_dns
logger.info(f"✓ Lakebase host: {lakebase_host}")

# COMMAND ----------

memory = LakebaseMemory(host=lakebase_host, instance_name=INSTANCE_NAME)
logger.info("✓ LakebaseMemory initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Search Retrieval

# COMMAND ----------


def retrieve_documents(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve relevant causal inference papers from vector search."""
    index_name = (
        f"{cfg.project.catalog}.{cfg.project.schema}.causal_inference_chunks_index"
    )
    index = vsc.get_index(index_name=index_name)

    results = index.similarity_search(
        query_text=query,
        columns=["text", "title", "paper_id", "summary"],
        num_results=num_results,
        query_type="hybrid",
    )

    documents = []
    if results and "result" in results:
        for row in results["result"].get("data_array", []):
            documents.append(
                {
                    "text": row[0],
                    "title": row[1],
                    "paper_id": row[2],
                    "summary": row[3],
                }
            )

    return documents


# Test retrieval
query = "causal identification strategies"
docs = retrieve_documents(query, num_results=3)
logger.info(f"Retrieved {len(docs)} documents for: '{query}'")
for i, doc in enumerate(docs, 1):
    logger.info(f"\n{i}. {doc['title']}")
    logger.info(f"   Paper ID: {doc['paper_id']}")
    logger.info(f"   Text preview: {doc['text'][:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. RAG Prompt Builder

# COMMAND ----------


def build_rag_prompt(question: str, documents: list[dict]) -> str:
    """Build a prompt enriched with retrieved paper context."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"Document {i}: {doc['title']}\n"
            f"Paper ID: {doc['paper_id']}\n"
            f"Content: {doc['text']}\n"
        )

    context = "\n---\n".join(context_parts)

    return (
        f"{cfg.project.system_prompt}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer based on the provided context\n"
        "- If the context lacks enough information, say so\n"
        "- Cite the paper title when making specific claims\n\n"
        "ANSWER:"
    )


# Test prompt building
test_prompt = build_rag_prompt("What is the difference-in-differences method?", docs)
logger.info(f"Prompt length: {len(test_prompt)} characters")
logger.info(f"Preview:\n{test_prompt[:400]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Single-Turn RAG Query

# COMMAND ----------


def rag_query(question: str, num_docs: int = 5) -> dict:
    """Answer a question using RAG (retrieve + generate)."""
    documents = retrieve_documents(question, num_results=num_docs)
    prompt = build_rag_prompt(question, documents)

    response = client.chat.completions.create(
        model=cfg.project.llm_endpoint,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.model.max_tokens,
        temperature=cfg.model.temperature,
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [{"title": d["title"], "paper_id": d["paper_id"]} for d in documents],
    }


result = rag_query("What are the main assumptions of the instrumental variables method?")
logger.info("=" * 80)
logger.info(f"Q: {result['question']}")
logger.info(f"\nA: {result['answer']}")
logger.info("\nSources:")
for src in result["sources"]:
    logger.info(f"  - {src['title']} ({src['paper_id']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Stateful SimpleRAG with LakebaseMemory
# MAGIC
# MAGIC Unlike the single-turn `rag_query` above, `SimpleRAG`:
# MAGIC - Loads conversation history from Lakebase on each turn
# MAGIC - Retrieves fresh documents for each question
# MAGIC - Appends the new exchange back to Lakebase
# MAGIC - Can resume a session across notebook restarts

# COMMAND ----------


class SimpleRAG:
    """Stateful RAG system backed by LakebaseMemory."""

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        index_name: str,
        memory: LakebaseMemory,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.index_name = index_name
        self.memory = memory

        self._w = WorkspaceClient()
        self._vsc = VectorSearchClient(
            workspace_url=self._w.config.host,
            personal_access_token=self._w.config.authenticate()["Authorization"].split(" ")[1],
            disable_notice=True,
        )

    def _retrieve(self, query: str, num_results: int = 5) -> list[dict]:
        index = self._vsc.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["text", "title", "paper_id"],
            num_results=num_results,
            query_type="hybrid",
        )
        docs = []
        if results and "result" in results:
            for row in results["result"].get("data_array", []):
                docs.append({"text": row[0], "title": row[1], "paper_id": row[2]})
        return docs

    def chat(self, session_id: str, question: str, num_docs: int = 3) -> str:
        """Chat with memory. Loads history, retrieves docs, generates, saves."""
        # Fresh client each call — OAuth tokens are short-lived
        llm_client = OpenAI(
            api_key=self._w.config.authenticate()["Authorization"].split(" ")[1],
            base_url=f"{self._w.config.host}/serving-endpoints",
        )
        history = self.memory.load_messages(session_id)

        docs = self._retrieve(question, num_results=num_docs)
        context = "\n\n".join(f"[{doc['title']}]: {doc['text']}" for doc in docs)

        system_msg = (
            f"{self.system_prompt}\n\n"
            f"CONTEXT FROM PAPERS:\n{context}\n\n"
            "If the context doesn't contain relevant information, say so. "
            "Always cite paper titles when making claims."
        )

        messages = (
            [{"role": "system", "content": system_msg}]
            + history
            + [{"role": "user", "content": question}]
        )

        response = llm_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content

        self.memory.save_messages(
            session_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )

        return answer


# COMMAND ----------

index_name = f"{cfg.project.catalog}.{cfg.project.schema}.causal_inference_chunks_index"
rag = SimpleRAG(
    llm_endpoint=cfg.project.llm_endpoint,
    system_prompt=cfg.project.system_prompt,
    index_name=index_name,
    memory=memory,
)

logger.info("✓ SimpleRAG initialized with LakebaseMemory")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Multi-Turn Conversation Demo

# COMMAND ----------

session_id = f"causal-inference-session-{uuid4()}"
logger.info(f"Session ID: {session_id}")

# COMMAND ----------

q1 = "What is the idea behind Grouped and Dummy Regression?"
a1 = rag.chat(session_id, q1)
logger.info(f"Q: {q1}")
logger.info(f"A: {a1}\n")

# COMMAND ----------

# Follow-up — uses conversation history automatically
q2 = "What are the key assumptions it relies on?"
a2 = rag.chat(session_id, q2)
logger.info(f"Q: {q2}")
logger.info(f"A: {a2}\n")

# COMMAND ----------

# Another follow-up
q3 = "How does it compare to instrumental variables?"
a3 = rag.chat(session_id, q3)
logger.info(f"Q: {q3}")
logger.info(f"A: {a3}")

# COMMAND ----------

# Verify history was persisted
history = memory.load_messages(session_id)
logger.info(f"✓ {len(history)} messages persisted for session {session_id}")
