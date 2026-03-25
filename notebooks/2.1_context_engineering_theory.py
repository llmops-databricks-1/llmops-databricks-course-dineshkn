# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.1: Context Engineering Theory
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is context engineering
# MAGIC - RAG (Retrieval Augmented Generation) fundamentals
# MAGIC - Context window limitations
# MAGIC - Strategies for effective context engineering
# MAGIC - Trade-offs and best practices

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Context Engineering?
# MAGIC
# MAGIC **Context Engineering** is the practice of providing relevant information to LLMs to improve their responses.
# MAGIC
# MAGIC ### Why Context Matters
# MAGIC
# MAGIC - LLMs have knowledge cutoff dates
# MAGIC - Need access to private/proprietary data
# MAGIC - Reduce hallucinations
# MAGIC - Improve accuracy and relevance
# MAGIC - Enable domain-specific applications
# MAGIC
# MAGIC For a causal inference knowledge base, this means grounding responses in the actual textbook content
# MAGIC rather than relying on the model's general knowledge of statistical methods.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RAG (Retrieval Augmented Generation)
# MAGIC
# MAGIC ### RAG Pipeline Overview
# MAGIC
# MAGIC ```
# MAGIC User Query
# MAGIC     ↓
# MAGIC Query Embedding
# MAGIC     ↓
# MAGIC Vector Search (Retrieve relevant document chunks)
# MAGIC     ↓
# MAGIC Context Assembly
# MAGIC     ↓
# MAGIC LLM Prompt (Query + Context)
# MAGIC     ↓
# MAGIC Generated Response
# MAGIC ```
# MAGIC
# MAGIC ### Benefits of RAG
# MAGIC
# MAGIC 1. **Up-to-date Information**: Access to latest data
# MAGIC 2. **Domain Knowledge**: Incorporate specialized information (e.g. causal inference textbooks)
# MAGIC 3. **Reduced Hallucinations**: Grounded in retrieved facts
# MAGIC 4. **Cost-Effective**: No need to fine-tune for every use case
# MAGIC 5. **Transparency**: Can cite sources

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Context Window Limitations
# MAGIC
# MAGIC ### Token Limits by Model
# MAGIC
# MAGIC | Model | Context Window | Notes |
# MAGIC |-------|---------------|-------|
# MAGIC | GPT-3.5 Turbo | 16K tokens | ~12K words |
# MAGIC | GPT-4 | 8K-128K tokens | Varies by version |
# MAGIC | Claude 3 | 200K tokens | ~150K words |
# MAGIC | Llama 3.1 70B | 128K tokens | ~96K words |
# MAGIC | Llama 3.1 405B | 128K tokens | ~96K words |
# MAGIC | Gemini 1.5 Pro | 1M tokens | ~750K words |

# COMMAND ----------

from loguru import logger

# COMMAND ----------

# Example: Token estimation
def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 characters per token."""
    return len(text) // 4


short_text = "What is an instrumental variable?"
long_text = (
    "Instrumental variables are used in causal inference to address endogeneity. "
    * 100
)

logger.info(f"Short text: {estimate_tokens(short_text)} tokens")
logger.info(f"Long text: {estimate_tokens(long_text)} tokens")

context_window = 128_000  # Llama 3.1
system_prompt_tokens = 200
user_query_tokens = 100
max_output_tokens = 2000

available_for_context = context_window - system_prompt_tokens - user_query_tokens - max_output_tokens
logger.info(f"\nAvailable tokens for context: {available_for_context:,}")
logger.info(f"Approximate words: {available_for_context * 0.75:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Retrieval Strategies Overview
# MAGIC
# MAGIC ### Three Main Approaches
# MAGIC
# MAGIC **1. Semantic Search (Vector Search)**
# MAGIC - Convert documents to embeddings (numerical vectors)
# MAGIC - Find documents with similar vector representations
# MAGIC - **Pros**: Captures meaning, handles synonyms
# MAGIC - **Cons**: May miss exact keyword matches
# MAGIC
# MAGIC **2. Hybrid Search**
# MAGIC - Combines semantic search (embeddings) + keyword search (BM25)
# MAGIC - **Pros**: Best of both worlds — meaning + exact matches (e.g. "DiD", "IV", "RDD")
# MAGIC - **Cons**: More complex, slightly slower
# MAGIC
# MAGIC **3. Reranking**
# MAGIC - Retrieve more candidates (e.g. top 20-50)
# MAGIC - Use a cross-encoder model to rerank
# MAGIC - **Pros**: Higher precision
# MAGIC - **Cons**: Additional computation cost
# MAGIC
# MAGIC **Note**: We'll implement these with actual code in **Notebook 2.4**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Enhancement Technique: Query Rewriting
# MAGIC
# MAGIC Before retrieval, enhance the query by generating variations:
# MAGIC - Use different terminology
# MAGIC - Make it more specific or general
# MAGIC - Focus on different aspects
# MAGIC
# MAGIC This improves recall by searching with multiple phrasings.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from openai import OpenAI
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints",
)

MODEL_NAME = cfg.project.llm_endpoint
logger.info(f"Using model: {MODEL_NAME}")

# COMMAND ----------


def rewrite_query(original_query: str) -> list[str]:
    """Generate query variations for better retrieval."""
    prompt = f"""Given this search query about causal inference, generate 3 alternative phrasings that would help retrieve relevant information:

Original query: {original_query}

Generate 3 variations that:
1. Use different terminology
2. Are more specific or more general
3. Focus on different aspects

Return only the 3 variations, one per line."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )

    variations = response.choices[0].message.content.strip().split("\n")
    return [v.strip() for v in variations if v.strip()]


# COMMAND ----------

original = "What is instrumental variable analysis?"
variations = rewrite_query(original)

logger.info(f"Original: {original}\n")
logger.info("Variations:")
for i, var in enumerate(variations, 1):
    logger.info(f"{i}. {var}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Context Quality vs Quantity
# MAGIC
# MAGIC ### The Trade-off
# MAGIC
# MAGIC - **More context** = More information, but:
# MAGIC   - Higher cost (more tokens)
# MAGIC   - Slower inference
# MAGIC   - Risk of "lost in the middle" problem
# MAGIC   - More noise
# MAGIC
# MAGIC - **Less context** = Faster and cheaper, but:
# MAGIC   - May miss important information
# MAGIC   - Less comprehensive answers

# COMMAND ----------


def order_context_by_relevance(chunks: list[dict]) -> list[dict]:
    """Order chunks to avoid 'lost in the middle' problem.

    Strategy: Most relevant at start, second-most at end, rest in middle.
    """
    if len(chunks) <= 2:
        return chunks

    ordered = [chunks[0]]
    if len(chunks) > 2:
        ordered.extend(chunks[2:-1])
    if len(chunks) > 1:
        ordered.append(chunks[1])

    return ordered


chunks = [
    {"text": "Instrumental variables require a valid instrument", "score": 0.95},
    {"text": "The exclusion restriction must hold", "score": 0.88},
    {"text": "Two-stage least squares is the standard estimator", "score": 0.75},
    {"text": "Weak instruments inflate standard errors", "score": 0.70},
    {"text": "The F-statistic tests instrument strength", "score": 0.65},
]

ordered = order_context_by_relevance(chunks)
logger.info("Ordered chunks (lost-in-the-middle strategy):")
for i, chunk in enumerate(ordered, 1):
    logger.info(f"{i}. {chunk['text']} (score: {chunk['score']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Context Compression Techniques
# MAGIC
# MAGIC ### Technique 1: Extractive Summarization
# MAGIC - Select most important sentences/chunks
# MAGIC - Fast, preserves original wording
# MAGIC
# MAGIC ### Technique 2: Abstractive Summarization
# MAGIC - Use LLM to generate summaries
# MAGIC - More concise and coherent
# MAGIC - Slower, costs tokens

# COMMAND ----------


def summarize_chunk(text: str, max_length: int = 100) -> str:
    """Summarize a text chunk using LLM."""
    prompt = f"""Summarize the following causal inference text in {max_length} words or less, preserving key concepts:

{text}

Summary:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length * 2,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


# COMMAND ----------

sample_text = """
Difference-in-differences (DiD) is a statistical technique used in econometrics and quantitative
research that attempts to mimic an experimental research design using observational study data,
by studying the differential effect of a treatment on a 'treatment group' versus a 'control group'
in a natural experiment. It calculates the effect of a treatment on an outcome by comparing the
average change over time in the outcome variable for the treatment group, compared to the average
change over time for the control group.
"""

summary = summarize_chunk(sample_text, max_length=50)
logger.info(f"Original length: {len(sample_text)} chars")
logger.info(f"Summary length: {len(summary)} chars")
logger.info(f"\nSummary:\n{summary}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Metadata Filtering
# MAGIC
# MAGIC Enhance retrieval with metadata filters:
# MAGIC
# MAGIC - **Chapter filters**: Search only specific chapters (e.g. only IV chapters)
# MAGIC - **Element type filters**: Prose vs code examples
# MAGIC - **Page range**: Narrow to specific sections

# COMMAND ----------

import json

example_document = {
    "id": "07 - Beyond Confounders_42",
    "text": "The FWL theorem states that...",
    "embedding": [0.1, 0.2, 0.3],
    "metadata": {
        "paper_id": "07 - Beyond Confounders — Causal Inference for the Brave and True",
        "chapter_num": "07",
        "title": "Beyond Confounders",
        "element_type": "PARAGRAPH",
        "page_number": 5,
    },
}

logger.info("Example chunk with metadata:")
logger.info(json.dumps(example_document, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Prompt Engineering for RAG
# MAGIC
# MAGIC ### Effective RAG Prompts
# MAGIC
# MAGIC ```
# MAGIC System: You are a helpful assistant specializing in causal inference.
# MAGIC Use the provided context to answer questions. If the answer is not
# MAGIC in the context, say "I don't have enough information to answer that."
# MAGIC
# MAGIC Context:
# MAGIC [Retrieved document chunks here]
# MAGIC
# MAGIC Question: [User question]
# MAGIC
# MAGIC Answer:
# MAGIC ```

# COMMAND ----------


def create_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """Create a RAG prompt with retrieved context."""
    context = "\n\n".join(
        [f"[Document {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    return f"""Use the following context to answer the question about causal inference. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""


query = "What is the difference between correlation and causation?"
context_chunks = [
    "Correlation measures the statistical association between two variables, but does not imply that one causes the other.",
    "Causation requires that changing one variable directly produces a change in another, which requires controlling for confounders.",
]

prompt = create_rag_prompt(query, context_chunks)
logger.info(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluation Metrics for Context Engineering
# MAGIC
# MAGIC ### Retrieval Metrics
# MAGIC - **Precision@K**: Proportion of retrieved chunks that are relevant
# MAGIC - **Recall@K**: Proportion of relevant chunks that are retrieved
# MAGIC - **MRR (Mean Reciprocal Rank)**: Position of first relevant result
# MAGIC - **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality
# MAGIC
# MAGIC ### End-to-End Metrics
# MAGIC - **Answer Relevance**: Is the answer relevant to the question?
# MAGIC - **Faithfulness**: Is the answer grounded in the context?
# MAGIC - **Context Relevance**: Is the retrieved context relevant?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Best Practices Summary
# MAGIC
# MAGIC ### ✅ Do:
# MAGIC 1. Chunk documents appropriately (more in next notebook)
# MAGIC 2. Use high-quality embeddings
# MAGIC 3. Implement metadata filtering (e.g. filter by chapter_num)
# MAGIC 4. Order context strategically (avoid lost-in-the-middle)
# MAGIC 5. Monitor retrieval quality
# MAGIC 6. Provide clear instructions in prompts
# MAGIC 7. Handle cases where context doesn't contain the answer
# MAGIC
# MAGIC ### ❌ Don't:
# MAGIC 1. Exceed context window limits
# MAGIC 2. Include irrelevant information
# MAGIC 3. Ignore the "lost in the middle" problem
# MAGIC 4. Forget to cite sources
# MAGIC 5. Assume all retrieved chunks are relevant
