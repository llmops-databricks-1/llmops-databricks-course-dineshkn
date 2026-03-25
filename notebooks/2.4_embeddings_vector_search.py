# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Embeddings & Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Understanding embeddings
# MAGIC - Different embedding models
# MAGIC - Creating vector search endpoints and indexes
# MAGIC - Querying with similarity search
# MAGIC - Advanced options: filters, hybrid search, reranking

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/dineshkaimal91@gmail.com/.bundle/llmops-databricks-course-dineshkn/dev/artifacts/.internal/causal_inference_curator-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from databricks.vector_search.reranker import DatabricksReranker
from loguru import logger
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config
from causal_inference_curator.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.project.catalog
schema = cfg.project.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Embeddings
# MAGIC
# MAGIC **Embeddings** are numerical representations of text that capture semantic meaning.
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC - **Vector**: Array of numbers (e.g., [0.1, -0.3, 0.5, ...])
# MAGIC - **Dimension**: Length of the vector (e.g., 384, 768, 1024)
# MAGIC - **Semantic Similarity**: Similar meanings = similar vectors
# MAGIC - **Distance Metrics**: Cosine similarity, Euclidean distance, dot product
# MAGIC
# MAGIC ### How it Works:
# MAGIC
# MAGIC ```
# MAGIC Text: "instrumental variables"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.23, -0.15, 0.67, ..., 0.42]  # 1024 dimensions
# MAGIC
# MAGIC Text: "IV estimation"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.25, -0.13, 0.65, ..., 0.40]  # Similar to above!
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Embedding Models Comparison
# MAGIC
# MAGIC | Model | Dimensions | Max Tokens | Best For |
# MAGIC |-------|-----------|------------|----------|
# MAGIC | **databricks-bge-large-en** | 1024 | 512 | General purpose, high quality |
# MAGIC | **databricks-gte-large-en** | 1024 | 512 | General purpose, fast |
# MAGIC | **text-embedding-ada-002** (OpenAI) | 1536 | 8191 | High quality, expensive |
# MAGIC | **e5-large-v2** | 1024 | 512 | Open source, good quality |
# MAGIC | **all-MiniLM-L6-v2** | 384 | 512 | Fast, smaller, lower quality |
# MAGIC
# MAGIC **For this course, we use `databricks-gte-large-en`** — fast, high-quality, and free on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Search Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Delta Table (causal_inference_chunks)    │
# MAGIC │  - id                                        │
# MAGIC │  - text                                      │
# MAGIC │  - chapter_num, title, paper_id              │
# MAGIC └──────────────┬───────────────────────────────┘
# MAGIC                │
# MAGIC                │ (Automatic sync via CDF)
# MAGIC                ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Vector Search Index                      │
# MAGIC │  - Embeddings generated automatically        │
# MAGIC │  - Stored in optimized format                │
# MAGIC │  - Supports similarity search                │
# MAGIC └──────────────┬───────────────────────────────┘
# MAGIC                │
# MAGIC                │ (Query)
# MAGIC                ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Search Results                           │
# MAGIC │  - Most similar chunks                       │
# MAGIC │  - With similarity scores                    │
# MAGIC └──────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Endpoint

# COMMAND ----------

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.project.vector_search_endpoint,
    embedding_model=cfg.project.embedding_endpoint,
)

logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model:        {vs_manager.embedding_model}")
logger.info(f"Index Name:             {vs_manager.index_name}")

# COMMAND ----------

vs_manager.create_endpoint_if_not_exists()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Endpoint Types:
# MAGIC
# MAGIC - **STANDARD**: General purpose, good performance
# MAGIC - **STANDARD_LARGE**: Higher throughput, more expensive
# MAGIC
# MAGIC For development and most production workloads, STANDARD is sufficient.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Vector Search Index

# COMMAND ----------

# Create or get the vector search index.
# This automatically:
# - Creates the index if it doesn't exist
# - Configures it with the embedding model (databricks-gte-large-en)
# - Sets up delta sync with the causal_inference_chunks table

index = vs_manager.create_or_get_index()

logger.info("\n✅ Vector search setup complete!")
logger.info(f"  Index:           {vs_manager.index_name}")
logger.info(f"  Source table:    {vs_manager.chunks_table}")
logger.info(f"  Embedding model: {vs_manager.embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Configuration Options:
# MAGIC
# MAGIC - **pipeline_type**:
# MAGIC   - `TRIGGERED`: Manual sync, good for batch processing
# MAGIC   - `CONTINUOUS`: Auto-sync with Change Data Feed, real-time updates
# MAGIC
# MAGIC - **primary_key**: Unique identifier for each chunk (`id`)
# MAGIC
# MAGIC - **embedding_source_column**: The text column to embed (`text`)
# MAGIC
# MAGIC - **embedding_model_endpoint_name**: Which embedding model to use

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Helper Function for Parsing Results

# COMMAND ----------


def parse_vector_search_results(results: dict) -> list[dict]:
    """Parse vector search results from array format to dict format.

    Args:
        results: Raw results from similarity_search()

    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in data_array]


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Semantic Search with Similarity
# MAGIC
# MAGIC ### How Semantic Search Works
# MAGIC
# MAGIC 1. **Query Embedding**: Convert your search query to a vector
# MAGIC 2. **Similarity Calculation**: Compare query vector to all chunk vectors using **cosine similarity**
# MAGIC 3. **Ranking**: Return chunks with highest similarity scores
# MAGIC
# MAGIC ### Cosine Similarity
# MAGIC
# MAGIC Measures the angle between two vectors (range: -1 to 1):
# MAGIC - **1.0**: Identical meaning
# MAGIC - **0.8-0.9**: Very similar
# MAGIC - **0.5-0.7**: Somewhat related
# MAGIC - **< 0.5**: Less relevant

# COMMAND ----------

query = "What is the difference between correlation and causation?"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "paper_id", "chapter_num"],
    num_results=5,
)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Chapter {row.get('chapter_num', 'N/A')}: {row.get('title', 'N/A')}")
    logger.info(f"   Chunk ID: {row.get('id', 'N/A')}")
    logger.info(f"   Text preview: {row.get('text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Advanced Search: Filters
# MAGIC
# MAGIC Filter search results by metadata to narrow the scope:

# COMMAND ----------

# Search within a specific chapter
query = "regression discontinuity design"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "chapter_num"],
    filters={"chapter_num": "07"},
    num_results=3,
)

logger.info(f"Query: {query}")
logger.info("Filter: chapter_num = '07'\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Chapter {row.get('chapter_num', 'N/A')}: {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter Examples:
# MAGIC
# MAGIC ```python
# MAGIC # Single chapter filter
# MAGIC filters = {"chapter_num": "07"}
# MAGIC
# MAGIC # Filter by element type (prose only)
# MAGIC filters = {"element_type": "PARAGRAPH"}
# MAGIC
# MAGIC # Filter by paper
# MAGIC filters = {"paper_id": "05 - The Unreasonable Effectiveness of Linear Regression..."}
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Hybrid Search: Semantic + Keyword
# MAGIC
# MAGIC ### Why Hybrid Search?
# MAGIC
# MAGIC **Semantic search alone** may miss:
# MAGIC - Exact technical terms (e.g. "2SLS", "DiD", "RDD")
# MAGIC - Acronyms and abbreviations common in causal inference
# MAGIC - Specific estimator names
# MAGIC
# MAGIC **Hybrid search** combines:
# MAGIC - **Semantic search** (embeddings) → Captures meaning, synonyms
# MAGIC - **Keyword search** (BM25) → Exact term matching, TF-IDF scoring
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. Run both searches in parallel
# MAGIC 2. Get top-k results from each
# MAGIC 3. **Fusion**: Merge using Reciprocal Rank Fusion (RRF)
# MAGIC 4. Return final top-k

# COMMAND ----------

query = "instrumental variables two stage least squares endogeneity"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "chapter_num"],
    num_results=5,
    query_type="hybrid",
)

logger.info(f"Query: {query}")
logger.info("Search Type: Hybrid (Semantic + Keyword)\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Chapter {row.get('chapter_num', 'N/A')}: {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Reranking for Higher Precision
# MAGIC
# MAGIC ### The Two-Stage Retrieval Pattern
# MAGIC
# MAGIC **Stage 1: Fast Retrieval** (Bi-encoder)
# MAGIC - Retrieve top 20-50 candidates quickly
# MAGIC - Uses pre-computed embeddings
# MAGIC - Fast but less accurate
# MAGIC
# MAGIC **Stage 2: Precise Reranking** (Cross-encoder)
# MAGIC - Score each candidate against the query
# MAGIC - More accurate relevance scoring
# MAGIC - Slower, but only runs on candidates
# MAGIC
# MAGIC | Aspect | Bi-encoder | Cross-encoder |
# MAGIC |--------|-----------|---------------|
# MAGIC | **Speed** | Very fast | Slower |
# MAGIC | **Accuracy** | Good | Excellent |
# MAGIC | **Use case** | Initial retrieval | Reranking |
# MAGIC | **How it works** | Separate query & doc embeddings | Joint query-doc encoding |

# COMMAND ----------

query = "difference in differences parallel trends assumption"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "chapter_num"],
    num_results=5,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "title"]),
)

logger.info(f"Query: {query}")
logger.info("With reranking on: text, title\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Chapter {row.get('chapter_num', 'N/A')}: {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Search Quality Comparison

# COMMAND ----------

query = "regression discontinuity local average treatment effect"

logger.info(f"Query: {query}\n")

# Strategy 1: Basic semantic search
results_basic = index.similarity_search(
    query_text=query,
    columns=["text", "title", "chapter_num"],
    num_results=3,
)

logger.info("Strategy 1: Basic Semantic Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_basic), 1):
    logger.info(f"{i}. Ch{row.get('chapter_num', '?')}: {row.get('title', 'N/A')[:60]}...")

# Strategy 2: Hybrid search
results_hybrid = index.similarity_search(
    query_text=query,
    columns=["text", "title", "chapter_num"],
    num_results=3,
    query_type="hybrid",
)

logger.info("\nStrategy 2: Hybrid Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_hybrid), 1):
    logger.info(f"{i}. Ch{row.get('chapter_num', '?')}: {row.get('title', 'N/A')[:60]}...")

# Strategy 3: Hybrid + Reranking
results_reranked = index.similarity_search(
    query_text=query,
    columns=["text", "title", "chapter_num"],
    num_results=3,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "title"]),
)

logger.info("\nStrategy 3: Hybrid + Reranking")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_reranked), 1):
    logger.info(f"{i}. Ch{row.get('chapter_num', '?')}: {row.get('title', 'N/A')[:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Monitoring and Maintenance

# COMMAND ----------

index_info = vs_manager.client.get_index(
    index_name=vs_manager.index_name,
)

logger.info("Index Information:")
logger.info(f"  Name:     {index_info.name}")
logger.info(f"  Endpoint: {index_info.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Maintenance:
# MAGIC
# MAGIC ```python
# MAGIC # Sync index manually (for TRIGGERED pipeline)
# MAGIC index.sync()
# MAGIC
# MAGIC # Delete index (if needed)
# MAGIC # vs_manager.client.delete_index(index_name=vs_manager.index_name)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ Understanding embeddings and vector representations
# MAGIC 2. ✅ Comparing different embedding models
# MAGIC 3. ✅ Creating vector search endpoints
# MAGIC 4. ✅ Creating and syncing vector search indexes
# MAGIC 5. ✅ Basic similarity search
# MAGIC 6. ✅ Advanced features: chapter filters, hybrid search, reranking
# MAGIC 7. ✅ Comparing search strategies
# MAGIC 8. ✅ Best practices and monitoring
# MAGIC
# MAGIC **Next**: Lecture 2.5 - Pipeline Design & Workflow
