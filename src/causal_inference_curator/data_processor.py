"""PDF processing pipeline for causal inference documents."""

from __future__ import annotations

import json
import re

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import col, concat_ws, explode, udf

from causal_inference_curator.config import Config

# Element types to exclude from chunks (binary/tabular content)
EXCLUDED_TYPES = {"table", "figure", "image"}


class DataProcessor:
    """Process causal inference PDFs from Unity Catalog Volume into a Delta chunks table.

    Pipeline:
    1. Read unprocessed papers from causal_inference_papers
    2. Parse PDFs using ai_parse_document (via SQL INSERT INTO)
    3. Extract and clean text chunks via UDF
    4. Join with paper metadata (title, summary, chapter_num)
    5. Write to causal_inference_chunks (CDF enabled)
    6. Update causal_inference_papers with processed timestamp and volume_path
    """

    def __init__(self, spark: SparkSession, config: Config) -> None:
        self.spark = spark
        self.config = config
        self.catalog = config.project.catalog
        self.schema = config.project.schema
        self.volume = config.project.volume
        self.papers_table = f"{self.catalog}.{self.schema}.causal_inference_papers"
        self.parsed_table = f"{self.catalog}.{self.schema}.ai_parsed_docs_table"
        self.chunks_table = f"{self.catalog}.{self.schema}.causal_inference_chunks"

    # ------------------------------------------------------------------
    # Static helpers (UDF-compatible: no self references)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
        """Extract text chunks from ai_parse_document JSON output.

        The JSON structure returned by ai_parse_document(content) is:
          {"document": {"elements": [{"type": "...", "id": "...", "content": "..."}]}}

        Excludes TABLE and FIGURE element types.
        """
        if not parsed_content_json:
            return []
        try:
            parsed = json.loads(parsed_content_json)
        except (json.JSONDecodeError, TypeError):
            return []

        chunks = []
        for element in parsed.get("document", {}).get("elements", []):
            element_type = element.get("type", "").lower()
            if element_type in EXCLUDED_TYPES:
                continue
            content = element.get("content", "").strip()
            if content:
                chunks.append((element.get("id", ""), content))
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalise chunk text."""
        # Fix hyphenation across line breaks: "docu-\nments" → "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)
        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _parse_pdfs_with_ai(self, volume_path: str) -> None:
        """Parse all PDFs in the volume using ai_parse_document; store JSON output."""
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path    STRING,
                parsed_content STRING,
                processed LONG
            )
        """)

        self.spark.sql(f"""
            INSERT INTO {self.parsed_table}
            SELECT
                path,
                ai_parse_document(content) AS parsed_content,
                unix_timestamp()            AS processed
            FROM READ_FILES(
                "{volume_path}",
                format          => 'binaryFile',
                pathGlobFilter  => '*.pdf'
            )
        """)
        logger.info(f"Parsed PDFs from {volume_path} into {self.parsed_table}")

    def _process_chunks(self) -> None:
        """Extract, clean, and join chunks; write to causal_inference_chunks."""
        chunk_schema = T.ArrayType(
            T.StructType(
                [
                    T.StructField("chunk_id", T.StringType(), True),
                    T.StructField("content", T.StringType(), True),
                ]
            )
        )
        extract_udf = udf(self._extract_chunks, chunk_schema)
        clean_udf = udf(self._clean_text, T.StringType())

        # Read latest parsed batch
        parsed_df = self.spark.table(self.parsed_table)

        # Build metadata lookup: paper_id → title, summary, chapter_num
        papers_meta = (
            self.spark.table(self.papers_table)
            .select("paper_id", "title", "summary")
            .withColumn(
                "chapter_num",
                F.regexp_extract(col("paper_id"), r"^(\d+)", 1),
            )
        )

        # Extract filename stem (paper_id) from path
        # binaryFile path: dbfs:/Volumes/catalog/schema/volume/05 - Title.pdf
        filename_stem = F.regexp_replace(
            F.element_at(F.split(col("path"), "/"), -1),
            r"\.pdf$",
            "",
        )

        chunks_df = (
            parsed_df.withColumn("paper_id", filename_stem)
            .withColumn("chunks", extract_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("paper_id"),
                col("path"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_udf(col("chunk.content")).alias("text"),
                concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias("id"),
            )
            .join(papers_meta, on="paper_id", how="left")
            .select(
                "id",
                "paper_id",
                "chapter_num",
                "title",
                "summary",
                "text",
                "path",
            )
        )

        # Ensure chunks table exists with Change Data Feed enabled
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                id           STRING,
                paper_id     STRING,
                chapter_num  STRING,
                title        STRING,
                summary      STRING,
                text         STRING
            )
            TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

        chunk_count = chunks_df.count()
        (
            chunks_df.drop("path")
            .write.format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .saveAsTable(self.chunks_table)
        )
        logger.info(f"✅ Written {chunk_count} chunks to {self.chunks_table}")

        # Update causal_inference_papers: set volume_path + processed timestamp
        volume_paths = (
            chunks_df.select("paper_id", "path")
            .distinct()
            .withColumn(
                "volume_path_clean",
                F.regexp_replace(col("path"), r"^dbfs:", ""),
            )
            .select("paper_id", "volume_path_clean")
        )
        volume_paths.createOrReplaceTempView("_processed_papers_tmp")

        self.spark.sql(f"""
            MERGE INTO {self.papers_table} AS target
            USING _processed_papers_tmp AS source
            ON target.paper_id = source.paper_id
            WHEN MATCHED THEN UPDATE SET
                target.volume_path = source.volume_path_clean,
                target.processed   = unix_timestamp()
        """)
        logger.info("✅ Updated causal_inference_papers with processed timestamps")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_and_save(self) -> None:
        """Run the full PDF processing pipeline."""
        logger.info("Starting PDF processing pipeline")

        unprocessed = (
            self.spark.table(self.papers_table).filter("processed IS NULL").count()
        )
        logger.info(f"Found {unprocessed} unprocessed papers")

        if unprocessed == 0:
            logger.info("No papers to process. Exiting.")
            return

        volume_path = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"
        self._parse_pdfs_with_ai(volume_path)
        self._process_chunks()
        logger.info("PDF processing pipeline complete")
