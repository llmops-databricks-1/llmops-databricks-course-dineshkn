# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.3: Session Memory with Lakebase
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Lakebase (Databricks managed PostgreSQL)
# MAGIC - Managing conversation history per session
# MAGIC - Connection pooling and authentication
# MAGIC - Building stateful agents

# COMMAND ----------

from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config
from causal_inference_curator.memory import LakebaseMemory

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create or Get Lakebase Instance
# MAGIC
# MAGIC **Lakebase** is Databricks' managed PostgreSQL:
# MAGIC - Fully managed and serverless
# MAGIC - Integrated with Databricks authentication (user tokens or SPN)
# MAGIC - Supports standard PostgreSQL features
# MAGIC - Ideal for session state and metadata

# COMMAND ----------

INSTANCE_NAME = "causal-inference-agent-instance"
USAGE_POLICY_ID = None  # Set to your usage policy ID if required by your workspace

try:
    instance = w.database.get_database_instance(INSTANCE_NAME)
    logger.info(f"✓ Using existing instance: {INSTANCE_NAME}")
    if instance.state == DatabaseInstanceState.STOPPED:
        logger.info("Instance stopped — restarting...")
        w.database.update_database_instance(
            name=INSTANCE_NAME,
            database_instance=DatabaseInstance(name=INSTANCE_NAME, stopped=False),
            update_mask="stopped",
        )
        instance = w.database.wait_get_database_instance_database_available(INSTANCE_NAME)
except Exception:
    logger.info(f"Creating new instance: {INSTANCE_NAME}")
    db_instance = DatabaseInstance(name=INSTANCE_NAME, capacity="CU_1")
    if USAGE_POLICY_ID:
        db_instance.usage_policy_id = USAGE_POLICY_ID
    instance = w.database.create_database_instance(db_instance)
    instance = w.database.wait_get_database_instance_database_available(INSTANCE_NAME)

lakebase_host = instance.read_write_dns
logger.info(f"✓ Lakebase host: {lakebase_host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Memory Manager

# COMMAND ----------

memory = LakebaseMemory(host=lakebase_host, instance_name=INSTANCE_NAME)
logger.info("✓ LakebaseMemory initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save and Load Messages

# COMMAND ----------

session_id = f"test-session-{uuid4()}"

test_messages = [
    {"role": "user", "content": "What papers discuss instrumental variables?"},
    {
        "role": "assistant",
        "content": "Several papers cover IV methods for causal identification...",
    },
    {"role": "user", "content": "Which one is most practical?"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"✓ Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

loaded = memory.load_messages(session_id)
logger.info(f"✓ Loaded {len(loaded)} messages:")
for i, msg in enumerate(loaded, 1):
    logger.info(f"  {i}. [{msg['role']}] {msg['content'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Multi-Turn Demo (save incrementally)

# COMMAND ----------

conversation_id = f"conversation-{uuid4()}"

memory.save_messages(
    conversation_id,
    [{"role": "user", "content": "I'm studying treatment effect estimation"}],
)
memory.save_messages(
    conversation_id,
    [
        {
            "role": "assistant",
            "content": "Treatment effect estimation covers ATE, ATT, and LATE...",
        }
    ],
)
memory.save_messages(conversation_id, [{"role": "user", "content": "What is LATE?"}])

full_conv = memory.load_messages(conversation_id)
logger.info(f"✓ Full conversation ({len(full_conv)} messages):")
for msg in full_conv:
    logger.info(f"  [{msg['role']}] {msg['content'][:80]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Using Memory with an LLM

# COMMAND ----------

client = OpenAI(
    api_key=w.config.authenticate()["Authorization"].split(" ")[1],
    base_url=f"{w.config.host}/serving-endpoints",
)


def chat_with_memory(session_id: str, user_message: str, memory: LakebaseMemory) -> str:
    """Chat with LLM using persisted session history."""
    previous = memory.load_messages(session_id)

    messages = (
        [{"role": "system", "content": cfg.project.system_prompt}]
        + previous
        + [{"role": "user", "content": user_message}]
    )

    response = client.chat.completions.create(
        model=cfg.project.llm_endpoint,
        messages=messages,
    )
    answer = response.choices[0].message.content

    memory.save_messages(
        session_id,
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ],
    )

    return answer


logger.info("✓ chat_with_memory ready")

# COMMAND ----------

agent_session = f"agent-session-{uuid4()}"

r1 = chat_with_memory(agent_session, "What is regression discontinuity design?", memory)
logger.info(f"Response 1: {r1[:200]}...")

# COMMAND ----------

r2 = chat_with_memory(agent_session, "What are its main limitations?", memory)
logger.info(f"Response 2: {r2[:200]}...")

# COMMAND ----------

# Verify full conversation is persisted
full = memory.load_messages(agent_session)
logger.info(f"✓ Full conversation ({len(full)} messages):")
for i, msg in enumerate(full, 1):
    content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
    logger.info(f"  {i}. [{msg['role']}] {content}")
