# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2: Model Context Protocol (MCP) Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is MCP?
# MAGIC - MCP vs custom functions
# MAGIC - Databricks Vector Search MCP server
# MAGIC - Creating MCP tools for agents
# MAGIC - Same agent loop — different tool source

# COMMAND ----------

import asyncio
import json

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from databricks_mcp import DatabricksMCPClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config
from causal_inference_curator.mcp import ToolInfo, create_mcp_tools
from causal_inference_curator.memory import LakebaseMemory

# Enable nested event loops (required in Databricks notebooks)
nest_asyncio.apply()

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Model Context Protocol (MCP)?
# MAGIC
# MAGIC **MCP** is a standardized protocol for connecting AI models to external tools.
# MAGIC
# MAGIC | Aspect | Custom Functions | MCP |
# MAGIC |--------|-----------------|-----|
# MAGIC | **Setup** | Write Python code | Use existing MCP server |
# MAGIC | **Maintenance** | You maintain | Databricks maintains |
# MAGIC | **Reusability** | Per-agent | Across agents |
# MAGIC | **Security** | Manual | Built-in |
# MAGIC | **Best For** | Custom logic | Standard operations |
# MAGIC
# MAGIC **Use MCP when**: You need standard operations (search, query, etc.)
# MAGIC **Use Custom Functions when**: You need custom business logic

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Databricks Vector Search MCP
# MAGIC
# MAGIC ### URL Format:
# MAGIC ```
# MAGIC {workspace_host}/api/2.0/mcp/vector-search/{catalog}/{schema}
# MAGIC ```
# MAGIC The server scans all vector search indexes in the catalog/schema and exposes
# MAGIC each one as a tool named `catalog__schema__index_name`.

# COMMAND ----------

host = w.config.host
vector_search_mcp_url = (
    f"{host}/api/2.0/mcp/vector-search/{cfg.project.catalog}/{cfg.project.schema}"
)
logger.info(f"Vector Search MCP URL:\n{vector_search_mcp_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. List Available MCP Tools

# COMMAND ----------

vs_mcp_client = DatabricksMCPClient(
    server_url=vector_search_mcp_url,
    workspace_client=w,
)

vs_tools = vs_mcp_client.list_tools()
logger.info(f"Vector Search MCP Tools ({len(vs_tools)}):")
for tool in vs_tools:
    logger.info(f"  Tool: {tool.name}")
    logger.info(f"  Description: {tool.description}")
    if tool.inputSchema:
        logger.info(
            f"  Parameters: {list(tool.inputSchema.get('properties', {}).keys())}"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Call Vector Search Tool Directly
# MAGIC
# MAGIC Tool name uses double underscores:
# MAGIC `mlops_dev__dineshka__causal_inference_chunks_index`
# MAGIC Only takes one parameter: `query`

# COMMAND ----------

tool_name = f"{cfg.project.catalog}__{cfg.project.schema}__causal_inference_chunks_index"

search_result = vs_mcp_client.call_tool(
    tool_name,
    {"query": "difference in differences causal inference"},
)

logger.info("Direct MCP Search Results:")
for content in search_result.content:
    logger.info(content.text[:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load MCP Tools via `create_mcp_tools()`

# COMMAND ----------

mcp_urls = [vector_search_mcp_url]

# Add Genie if configured
if cfg.project.genie_space_id:
    mcp_urls.append(f"{host}/api/2.0/mcp/genie/{cfg.project.genie_space_id}")

logger.info(f"Loading tools from {len(mcp_urls)} MCP server(s)...")
mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

logger.info(f"✓ Loaded {len(mcp_tools)} MCP tools:")
for i, tool in enumerate(mcp_tools, 1):
    logger.info(f"  {i}. {tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. View MCP Tool Specifications (what the LLM sees)

# COMMAND ----------

for tool in mcp_tools[:2]:
    logger.info(f"Tool: {tool.name}")
    logger.info(json.dumps(tool.spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Troubleshooting — Test MCP Connection

# COMMAND ----------


def test_mcp_connection(mcp_url: str) -> bool:
    """Test if MCP server is accessible."""
    try:
        c = DatabricksMCPClient(server_url=mcp_url, workspace_client=w)
        tools = c.list_tools()
        logger.info(f"✓ Connected — {len(tools)} tool(s) available")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return False


test_mcp_connection(vector_search_mcp_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. SimpleAgent Using MCP Tools + LakebaseMemory
# MAGIC
# MAGIC The agent loop is **identical** to 3.1b.
# MAGIC The only difference: tools come from `create_mcp_tools()` instead of `ToolRegistry`.

# COMMAND ----------

INSTANCE_NAME = "causal-inference-agent-instance"

try:
    instance = w.database.get_database_instance(INSTANCE_NAME)
    if instance.state == DatabaseInstanceState.STOPPED:
        w.database.update_database_instance(
            name=INSTANCE_NAME,
            database_instance=DatabaseInstance(name=INSTANCE_NAME, stopped=False),
            update_mask="stopped",
        )
        instance = w.database.wait_get_database_instance_database_available(INSTANCE_NAME)
except Exception:
    instance = w.database.create_database_instance(
        DatabaseInstance(name=INSTANCE_NAME, capacity="CU_1")
    )
    instance = w.database.wait_get_database_instance_database_available(INSTANCE_NAME)

memory = LakebaseMemory(
    host=instance.read_write_dns,
    instance_name=INSTANCE_NAME,
)
logger.info("✓ LakebaseMemory ready")

# COMMAND ----------


class SimpleAgent:
    """Agent using MCP tools + LakebaseMemory (identical loop to 3.1b)."""

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo],
        memory: LakebaseMemory,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.memory = memory
        self._tools_dict = {tool.name: tool for tool in tools}
        self._client = OpenAI(
            api_key=w.config.authenticate()["Authorization"].split(" ")[1],
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, session_id: str, user_message: str, max_iterations: int = 10) -> str:
        """Stateful chat: loads history, runs tool loop, saves history."""
        history = self.memory.load_messages(session_id)

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + history
            + [{"role": "user", "content": user_message}]
        )

        for _ in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )
            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"[MCP] Calling tool: {tool_name}({tool_args})")
                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {e}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                answer = assistant_message.content
                self.memory.save_messages(
                    session_id,
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": answer},
                    ],
                )
                return answer

        return "Max iterations reached."


# COMMAND ----------

from uuid import uuid4

agent = SimpleAgent(
    llm_endpoint=cfg.project.llm_endpoint,
    system_prompt=cfg.project.system_prompt,
    tools=mcp_tools,
    memory=memory,
)

logger.info("✓ MCP Agent created:")
for name in agent._tools_dict:
    logger.info(f"  - {name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Side-by-Side Comparison: Custom vs MCP (same query)

# COMMAND ----------

mcp_session = f"mcp-session-{uuid4()}"

q = "What is the main idea behind The Unreasonable Effectiveness of Linear Regression?"
logger.info(f"Query: {q}")
logger.info("=" * 80)

response = agent.chat(mcp_session, q)
logger.info(f"MCP Agent response:\n{response}")

# COMMAND ----------

# Show that history is persisted just like in 3.1b
history = memory.load_messages(mcp_session)
logger.info(f"✓ {len(history)} messages saved to Lakebase for session {mcp_session}")
