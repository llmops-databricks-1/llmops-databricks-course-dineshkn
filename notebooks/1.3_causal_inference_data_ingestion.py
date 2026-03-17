# Databricks notebook source
"""
Notebook 1.3 — Causal Inference PDF Data Ingestion

Overview:
    This notebook processes local PDF files about causal inference
    and stores their metadata as a Delta table in Unity Catalog.

Steps:
    1. Load environment configuration (catalog, schema, endpoints)
       from project_config.yml.
    2. Extract metadata from PDF files (title, page count, file size, text content).
    3. Create a Spark DataFrame with a defined schema and write it to a Delta table
       in Unity Catalog ({catalog}.{schema}.causal_inference_papers).
    4. Verify the ingested data by printing the schema, record count, and sample rows.
    5. Compute data statistics: file sizes, page counts, etc.

Output:
    Delta table: {catalog}.{schema}.causal_inference_papers
"""

import random
from datetime import datetime
from pathlib import Path

from loguru import logger
from pypdf import PdfReader
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

from causal_inference_curator.config import get_env, load_config

# COMMAND ----------
# Create Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Get environment and load config
env = get_env(spark)
logger.info(f"Environment: {env}")

# COMMAND ----------
cfg = load_config("../project_config.yml", env)
logger.info(f"Config loaded for catalog: {cfg.project.catalog}")

# COMMAND ----------
# Data management
CATALOG = cfg.project.catalog
SCHEMA = cfg.project.schema
TABLE_NAME = "causal_inference_papers"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} is ready.")

# COMMAND ----------


def fetch_pdf_metadata(pdf_dir: str = "../") -> list[dict]:
    """Extract metadata from local PDF files.

    Args:
        pdf_dir: Directory containing PDF files (default: parent directory)

    Returns:
        List of paper metadata dictionaries

    Example:
        >>> papers = fetch_pdf_metadata("../")
        >>> papers[0]["title"]
        '05 - The Unreasonable Effectiveness of Linear Regression'
        >>> papers[0]["page_count"]
        42
    """
    pdf_path_obj = Path(pdf_dir)
    pdf_files = list(pdf_path_obj.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []

    papers = []

    for pdf_file in pdf_files:
        try:
            with open(pdf_file, "rb") as f:
                reader = PdfReader(f)

                # Extract text from first few pages for summary
                summary_text = ""
                for page_num in range(min(3, len(reader.pages))):
                    summary_text += reader.pages[page_num].extract_text()

                # Trim summary to reasonable length
                summary = summary_text[:1000] if summary_text else None

                # Parse title from filename (remove chapter number and extension)
                title = pdf_file.stem

                paper = {
                    "paper_id": pdf_file.stem,
                    "title": title,
                    "summary": summary,
                    "page_count": len(reader.pages),
                    "file_size_bytes": pdf_file.stat().st_size,
                    "file_path": str(pdf_file.absolute()),
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "processed": None,
                    "volume_path": None,
                }

                papers.append(paper)
                logger.info(f"Processed: {title} ({len(reader.pages)} pages)")

        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")
            continue

    return papers


# COMMAND ----------
# Fetch PDF metadata

logger.info("Extracting PDF metadata...")
volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{cfg.project.volume}"
papers = fetch_pdf_metadata(volume_path)
logger.info(f"Extracted metadata from {len(papers)} PDFs")

# COMMAND ----------
# Show sample paper
if papers:
    sample_number = random.randint(0, len(papers) - 1)

    logger.info(f"Sample paper (index {sample_number}):")
    logger.info(f"Title: {papers[sample_number]['title']}")
    logger.info(f"Pages: {papers[sample_number]['page_count']}")
    logger.info(f"Size: {papers[sample_number]['file_size_bytes']} bytes")
    logger.info(f"Paper ID: {papers[sample_number]['paper_id']}")

# COMMAND ----------
# Create Delta Table in Unity Catalog

schema = StructType(
    [
        StructField("paper_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("summary", StringType(), True),
        StructField("page_count", LongType(), True),
        StructField("file_size_bytes", LongType(), True),
        StructField("file_path", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),
        StructField("volume_path", StringType(), True),
    ]
)

# Create Spark DataFrame
df = spark.createDataFrame(papers, schema=schema)

# COMMAND ----------
# Write to Delta table
output_table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

(
    df.write.format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(output_table_path)
)

logger.info(f"Created Delta Table: {output_table_path}")
logger.info(f"Records: {df.count()}")

# COMMAND ----------
# Verify the data

papers_df = spark.table(output_table_path)
logger.info(f"Table: {output_table_path}")
logger.info(f"Total papers: {papers_df.count()}")
logger.info("Table Schema:")
papers_df.printSchema()

# Sample records
logger.info("Sample records:")
papers_df.select("paper_id", "title", "page_count", "file_size_bytes").show(
    5, truncate=50
)

# COMMAND ----------
# Data Statistics

logger.info("PDF Statistics:")
papers_df.select("page_count", "file_size_bytes").describe().show()

logger.info("Papers sorted by page count:")
papers_df.select("title", "page_count").orderBy("page_count", ascending=False).show(
    5, truncate=60
)

logger.info("Papers sorted by file size:")
papers_df.select("title", "file_size_bytes").orderBy(
    "file_size_bytes", ascending=False
).show(5, truncate=60)

# COMMAND ----------
