# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Custom Functions & Tools for Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What are agent tools?
# MAGIC - Creating custom functions
# MAGIC - Tool specifications (OpenAI format)
# MAGIC - Integrating tools with agents
# MAGIC - Vector search as a tool

# COMMAND ----------

import json
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config
from causal_inference_curator.mcp import ToolInfo

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tool Specification Format
# MAGIC
# MAGIC Tools are defined using the **OpenAI function calling format**:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "type": "function",
# MAGIC   "function": {
# MAGIC     "name": "tool_name",
# MAGIC     "description": "What the tool does",
# MAGIC     "parameters": {
# MAGIC       "type": "object",
# MAGIC       "properties": {
# MAGIC         "param1": {"type": "string", "description": "..."}
# MAGIC       },
# MAGIC       "required": ["param1"]
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Search Tool — `search_papers`

# COMMAND ----------


def parse_vector_search_results(results: dict) -> list[dict]:
    """Parse vector search results from array format to dict format."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row)) for row in data_array]


def search_papers(
    query: str, num_results: int = 5, year_filter: str | None = None
) -> str:
    """Search causal inference research papers using semantic vector search.

    Args:
        query: Search query describing what papers to find
        num_results: Number of results to return (default: 5)
        year_filter: Optional year filter (e.g., "2024")

    Returns:
        JSON string with search results containing title, paper_id, and excerpt
    """
    index_name = (
        f"{cfg.project.catalog}.{cfg.project.schema}.causal_inference_chunks_index"
    )
    index = vsc.get_index(index_name=index_name)

    search_params: dict[str, Any] = {
        "query_text": query,
        "columns": ["text", "title", "paper_id", "summary"],
        "num_results": num_results,
        "query_type": "hybrid",
    }

    if year_filter:
        search_params["filters"] = {"year": year_filter}

    results = index.similarity_search(**search_params)

    papers = []
    for row in parse_vector_search_results(results):
        papers.append(
            {
                "title": row.get("title", "N/A"),
                "paper_id": row.get("paper_id", "N/A"),
                "excerpt": row.get("text", "")[:300] + "...",
                "summary": row.get("summary", "")[:150],
            }
        )

    return json.dumps(papers, indent=2)


# Test the function
results = search_papers("instrumental variables", num_results=2)
logger.info("Search Results:")
logger.info(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tool Specification

# COMMAND ----------

search_papers_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": (
            "Search causal inference research papers using semantic similarity. "
            "Use this to find relevant papers before answering any research question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what papers to find",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
                "year_filter": {
                    "type": "string",
                    "description": "Optional year filter to limit results (e.g., '2024')",
                },
            },
            "required": ["query"],
        },
    },
}

logger.info("Search Papers Tool Specification:")
logger.info(json.dumps(search_papers_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ToolInfo and ToolRegistry

# COMMAND ----------

search_papers_tool = ToolInfo(
    name="search_papers",
    spec=search_papers_tool_spec,
    exec_fn=search_papers,
)

logger.info(f"Tool created: {search_papers_tool.name}")

# COMMAND ----------


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        self._tools[tool.name] = tool
        logger.info(f"✓ Registered tool: {tool.name}")

    def get_tool(self, name: str) -> ToolInfo:
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        return self.get_tool(name).exec_fn(**args)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> list[ToolInfo]:
        return list(self._tools.values())


registry = ToolRegistry()
registry.register(search_papers_tool)
logger.info(f"Tools registered: {registry.list_tools()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test the Tool

# COMMAND ----------

test_queries = [
    {"query": "instrumental variables causal identification"},
    {"query": "difference in differences policy evaluation"},
    {"query": "randomized controlled trials treatment effects"},
]

for tc in test_queries:
    logger.info(f"Query: {tc['query']}")
    result = registry.execute("search_papers", tc)
    logger.info(f"Result preview: {result[:150]}...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Best Practices for Tool Design
# MAGIC
# MAGIC **Do:**
# MAGIC 1. Write clear descriptions — the LLM reads them to decide when to call the tool
# MAGIC 2. Use proper Python type hints
# MAGIC 3. Return structured data (JSON string or clear text)
# MAGIC 4. Validate inputs before execution
# MAGIC
# MAGIC **Don't:**
# MAGIC 1. Create tools that are too complex or overlap
# MAGIC 2. Return unstructured or ambiguous data
# MAGIC 3. Make tools that take a long time to execute
# MAGIC 4. Forget error handling

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. SimpleAgent — LLM + Tool Loop

# COMMAND ----------


class SimpleAgent:
    """A simple agent that calls tools in a loop until the LLM stops requesting them."""

    def __init__(
        self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self._w = w
        self._client = None  # created fresh per chat() call to avoid token expiry

    def get_tool_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Single-turn chat. Memory is NOT persisted here — see 3.1b for that."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_iterations):
            self._client = OpenAI(
                api_key=self._w.config.authenticate()["Authorization"].split(" ")[1],
                base_url=f"{self._w.config.host}/serving-endpoints",
            )
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
                    logger.info(f"Calling tool: {tool_name}({tool_args})")
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
                return assistant_message.content

        return "Max iterations reached."


# COMMAND ----------

agent = SimpleAgent(
    llm_endpoint=cfg.project.llm_endpoint,
    system_prompt=cfg.project.system_prompt,
    tools=registry.get_all_tools(),
)

logger.info("✓ Agent created. Testing...")
response = agent.chat("What methods do papers use to estimate causal effects?")
logger.info(f"Agent response:\n{response}")
