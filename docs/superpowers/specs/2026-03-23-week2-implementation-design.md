# Week 2 Implementation Design — Causal Inference RAG Pipeline

**Date:** 2026-03-23
**Branch:** week2
**Course:** LLMOps on Databricks

---

## Overview

Implement Week 2 of the LLMOps course, adapting the reference `arxiv-curator` notebooks and utility modules for the causal inference use-case. The corpus is a set of causal inference PDF textbook chapters ("Causal Inference for the Brave and True") already stored in a Unity Catalog Volume and registered in the `causal_inference_papers` Delta table from Week 1.

The implementation covers:
- Two new utility modules in `src/causal_inference_curator/`
- Four new notebooks in `notebooks/`

All code runs on Databricks Serverless (directly or via VS Code + Databricks extension). No unit tests for Databricks-dependent modules.

---

## Critical Design Notes

### Config Access Pattern

The course project's `load_config()` returns a nested `Config` object — **not** a flat `ProjectConfig` like the reference. Always use `cfg.project.xxx`:

```python
# Reference (arxiv_curator) — WRONG pattern for this project
catalog = cfg.catalog
schema = cfg.schema

# This project — CORRECT
catalog = cfg.project.catalog
schema = cfg.project.schema
cfg.project.llm_endpoint
cfg.project.embedding_endpoint
cfg.project.vector_search_endpoint
```

Both `DataProcessor` and `VectorSearchManager` accept `Config` (not `ProjectConfig`).

### Filename Stem Extraction & Path Normalisation

`spark.read.format("binaryFile")` on Databricks returns paths with the `dbfs:` prefix, e.g.:
```
dbfs:/Volumes/mlops_dev/dineshka/causal_inference_pdfs/05 - The Unreasonable....pdf
```

`causal_inference_papers.paper_id` is the plain filename stem (no prefix, no extension), e.g.:
```
05 - The Unreasonable Effectiveness of Linear Regression — Causal Inference for the Brave and True
```

In `DataProcessor`, extract the stem from the binary path using:
```python
from pyspark.sql.functions import regexp_extract, col
# Strip directory prefix and .pdf extension to get stem
col("path").contains("/").cast(...)  # use regexp_replace or split
# Recommended: F.regexp_replace(F.element_at(F.split(col("path"), "/"), -1), r"\.pdf$", "")
```

When updating `volume_path` in `causal_inference_papers`, store the clean `/Volumes/...` path (without `dbfs:` prefix), matching the format used in `file_path` from Week 1.

### chapter_num Zero-Padding

`chapter_num` is stored as STRING. All filenames in the corpus use zero-padded two-digit prefixes (`"05"`, `"06"`, `"07"`). This is required for correct lexicographic ordering in Vector Search range filters. Extract using:
```python
F.regexp_extract(filename_col, r"^(\d+)", 1)
```

### SparkSession Pattern

All notebooks use `SparkSession.builder.getOrCreate()` — consistent with Week 1 (notebook 1.3). Do **not** use `DatabricksSession.builder.getOrCreate()`. The VS Code + Databricks extension transparently routes Spark calls to the remote cluster.

---

## Source Modules

### `src/causal_inference_curator/data_processor.py`

**Class:** `DataProcessor(spark: SparkSession, config: Config)`

**Attributes:**
- `spark`, `config`
- `catalog`, `schema`, `volume` (from `config.project`)
- `papers_table`: `{catalog}.{schema}.causal_inference_papers`
- `chunks_table`: `{catalog}.{schema}.causal_inference_chunks`

**Method: `process_and_save()`**

Full pipeline:
1. Load unprocessed papers from `causal_inference_papers` where `processed IS NULL`
2. Read PDFs as binary files: `spark.read.format("binaryFile").load(f"/Volumes/{catalog}/{schema}/{volume}/*.pdf")`
3. Parse each PDF using `ai_parse_document(content, named_struct('mode', 'DOCUMENT_INTELLIGENCE'))` via `expr()`
4. Explode the `elements` array; filter out `TABLE` and `FIGURE` element types; drop null/empty text
5. Extract `chapter_num` via `regexp_extract` on filename stem
6. Join with `causal_inference_papers` on filename stem (`paper_id`) to pull in `title`, `summary`
7. Assign `chunk_index` per document using `row_number()` window partitioned by `paper_id`
8. Assign `id` = `concat(paper_id, "_", chunk_index)`
9. Ensure `causal_inference_chunks` table exists with CDF enabled:
   ```sql
   CREATE TABLE IF NOT EXISTS {chunks_table} (...)
   TBLPROPERTIES (delta.enableChangeDataFeed = true)
   ```
10. Write chunks to `causal_inference_chunks` (mode: `overwrite`, `mergeSchema=true`)
11. UPDATE `causal_inference_papers`: set `volume_path = regexp_replace(path, '^dbfs:', '')`, `processed = unix_timestamp()` for processed papers — matched on `paper_id`

**Chunks table schema:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | STRING | `{paper_id}_{chunk_index}` — primary key for vector search |
| `paper_id` | STRING | Filename stem (matches `causal_inference_papers.paper_id`) |
| `chapter_num` | STRING | Zero-padded numeric prefix (e.g. `"05"`) |
| `title` | STRING | Full paper title from papers table |
| `summary` | STRING | First-pages text from papers table |
| `text` | STRING | Chunk text from AI Parse |
| `chunk_index` | INT | Position within document |
| `page_number` | INT | Page number from AI Parse |
| `element_type` | STRING | E.g. PARAGRAPH, CODE, HEADING |

---

### `src/causal_inference_curator/vector_search.py`

**Class:** `VectorSearchManager(config: Config, endpoint_name: str, embedding_model: str)`

**Attributes:**
- `endpoint_name`, `embedding_model`
- `catalog`, `schema` (from `config.project`)
- `index_name`: `{catalog}.{schema}.causal_inference_chunks_index`
- `chunks_table`: `{catalog}.{schema}.causal_inference_chunks`
- `client`: `VectorSearchClient()` instance

**Methods:**

**`create_endpoint_if_not_exists()`**
- List endpoints via `client.list_endpoints()`; if `endpoint_name` not present, call `client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")`
- Poll with `time.sleep(30)` until endpoint state is `ONLINE`; raise `RuntimeError` after 20 minutes

**`create_or_get_index()` → index object**
- Try `client.create_delta_sync_index(endpoint_name=..., index_name=..., source_table_name=chunks_table, pipeline_type="TRIGGERED", primary_key="id", embedding_source_column="text", embedding_model_endpoint_name=embedding_model)`
- If the index already exists (catch `Exception` checking for "already exists" in message), fall back to `client.get_index(endpoint_name=..., index_name=...)`
- Return the index object

**Package imports:** Both modules are accessed via direct imports (`from causal_inference_curator.data_processor import DataProcessor`). No changes to `__init__.py` are required — the modules just need to be present in the package directory.

---

## Notebooks

All notebooks:
- Start with `# Databricks notebook source`
- Use `# COMMAND ----------` cell separators
- Use `SparkSession.builder.getOrCreate()`
- Load config with `cfg = load_config("../project_config.yml", env)` and access via `cfg.project.xxx`

### `2.1_context_engineering_theory.py`
Pure theory/markdown with minimal runnable code cells. Covers: context engineering concepts, RAG pipeline overview, context window limits, retrieval strategies, query rewriting, context quality vs quantity, compression techniques, metadata filtering, prompt engineering for RAG, evaluation metrics, best practices.

Runnable cells use causal inference examples:
- Query rewriting: `"What is instrumental variable analysis?"`
- LLM model: `cfg.project.llm_endpoint` (not hardcoded string)
- Connect via `WorkspaceClient` + `OpenAI` client (same pattern as reference)

**Can run locally via VS Code + Databricks extension:** Yes (requires live cluster for LLM calls).

---

### `2.2_pdf_parsing_ai_parse.py`
Thin notebook. Imports `DataProcessor`, instantiates it, calls `processor.process_and_save()`. Explains the pipeline in markdown (comparison table of PDF parsing tools, pipeline step breakdown). The `%pip install` line stays commented out — wheel is deployed via bundle.

**Why it cannot run as a plain local Python script:** `ai_parse_document` is a Databricks SQL function unavailable outside a Databricks cluster.

**Can run locally via VS Code + Databricks extension:** Yes.

---

### `2.3_chunking_strategies.py`
Loads `causal_inference_chunks` table (exact reference: `spark.table(f"{catalog}.{schema}.causal_inference_chunks")`), shows chunk statistics (count, avg/min/max length), demonstrates alternative chunking strategies (fixed-size with overlap, sentence-based). Contains `%pip install` and `%restart_python` magic cells (same as reference).

**Why it cannot run as a plain local Python script:** `%restart_python` is a Databricks-only magic command that restarts the Python interpreter on the cluster kernel; it has no equivalent in a local Jupyter/script context.

**Can run locally via VS Code + Databricks extension:** Yes (magic cells are handled by the extension).

---

### `2.4_embeddings_vector_search.py`
Imports `VectorSearchManager`. Covers: embeddings theory, embedding model comparison, vector search architecture, endpoint/index creation, basic similarity search, filtered search (`chapter_num`), hybrid search, reranking comparison.

All queries are causal inference themed:
- `"What is the difference between correlation and causation?"`
- `"How do instrumental variables address endogeneity?"`
- `"Explain the difference-in-differences estimation method"`

Filter demo: `{"chapter_num": "07"}`.
Columns in search results: `["text", "id", "title", "paper_id", "chapter_num"]`.

**Can run locally via VS Code + Databricks extension:** Yes. `VectorSearchClient` calls go to the hosted Databricks Vector Search service.

---

## Key Differences from Reference

| Aspect | Reference (`arxiv-curator`) | This project |
|--------|---------------------------|--------------|
| Package | `arxiv_curator` | `causal_inference_curator` |
| Config access | `cfg.catalog` | `cfg.project.catalog` |
| Papers table | `arxiv_papers` | `causal_inference_papers` |
| Chunks table | `arxiv_chunks_table` | `causal_inference_chunks` (no `_table` suffix) |
| VS index | `arxiv_chunks_index` | `causal_inference_chunks_index` |
| Extra metadata | `arxiv_id`, `authors`, `year` | `chapter_num` (zero-padded STRING) |
| PDF source | Downloaded from arXiv API | Already in Unity Catalog Volume |
| Query examples | ML/NLP research topics | Causal inference concepts |
| Filter field | `year` | `chapter_num` |
| Session type | `DatabricksSession` (in 2.2) | `SparkSession` throughout |

---

## Files Created

```
src/causal_inference_curator/
├── data_processor.py          (new)
└── vector_search.py           (new)

notebooks/
├── 2.1_context_engineering_theory.py    (new)
├── 2.2_pdf_parsing_ai_parse.py          (new)
├── 2.3_chunking_strategies.py           (new)
└── 2.4_embeddings_vector_search.py      (new)

docs/superpowers/specs/
└── 2026-03-23-week2-implementation-design.md  (this file)
```

No changes to `pyproject.toml`, `project_config.yml`, `databricks.yml`, or `__init__.py` required.
