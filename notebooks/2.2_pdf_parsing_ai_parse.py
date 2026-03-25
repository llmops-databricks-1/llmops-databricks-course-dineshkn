# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: PDF Parsing with AI Parse Documents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - PDF parsing tools comparison
# MAGIC - Databricks AI Parse Documents for intelligent parsing
# MAGIC - Full pipeline: parse → chunk → store in Delta table
# MAGIC - Change Data Feed for real-time updates

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/dineshkaimal91@gmail.com/.bundle/llmops-databricks-course-dineshkn/dev/artifacts/.internal/causal_inference_curator-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config
from causal_inference_curator.data_processor import DataProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
logger.info("✅ Spark session ready")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

processor = DataProcessor(spark=spark, config=cfg)

logger.info(f"Catalog: {cfg.project.catalog}")
logger.info(f"Schema:  {cfg.project.schema}")
logger.info(f"Volume:  {cfg.project.volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. PDF Parsing Tools Comparison
# MAGIC
# MAGIC | Tool | Pros | Cons | Best For |
# MAGIC |------|------|------|----------|
# MAGIC | **AI Parse Documents** | - AI-powered<br>- Handles complex layouts<br>- Integrated with Databricks<br>- Preserves structure | - Databricks-specific<br>- Cost per page | Complex documents, code blocks, multi-column |
# MAGIC | **PyPDF2** | - Simple<br>- Free<br>- Pure Python | - Poor with complex layouts<br>- No structure extraction | Simple text extraction |
# MAGIC | **pdfplumber** | - Good table extraction<br>- Layout analysis | - Slower<br>- Manual tuning needed | Tables and structured data |
# MAGIC | **Apache Tika** | - Multi-format support<br>- Metadata extraction | - Java dependency<br>- Heavy | Multi-format processing |
# MAGIC | **Unstructured.io** | - ML-powered<br>- Good chunking | - External service<br>- API costs | Modern RAG pipelines |
# MAGIC
# MAGIC **AI Parse Documents** is the recommended choice for Databricks users — it understands document
# MAGIC structure (headings, paragraphs, code blocks) and integrates natively with Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parse PDFs and Create Chunks
# MAGIC
# MAGIC The `DataProcessor` class from `causal_inference_curator.data_processor` handles the full pipeline:
# MAGIC
# MAGIC 1. Load unprocessed papers from `causal_inference_papers` (where `processed IS NULL`)
# MAGIC 2. Read PDFs as binary files from the Unity Catalog Volume
# MAGIC 3. Parse each PDF using `ai_parse_document(content, named_struct('mode', 'DOCUMENT_INTELLIGENCE'))`
# MAGIC 4. Explode elements; filter out TABLE and FIGURE types; drop empty text
# MAGIC 5. Extract `chapter_num` from filename prefix (e.g. `"07"` from `"07 - Beyond Confounders..."`)
# MAGIC 6. Join with `causal_inference_papers` to attach `title` and `summary`
# MAGIC 7. Write chunks to `causal_inference_chunks` Delta table with Change Data Feed enabled
# MAGIC 8. Update `causal_inference_papers` with `processed` timestamp and `volume_path`
# MAGIC
# MAGIC The same class is used in `resources/` job definitions for scheduled runs.

# COMMAND ----------

processor.process_and_save()
