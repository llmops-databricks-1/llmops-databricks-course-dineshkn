# LLMOps Course — Learnings

## Week 1: Databricks Setup & Data Ingestion

### Workspace

- Started on **Databricks Free Edition** (`dbc-9a55c4a7-8e53`), moved to a paid workspace (`dbc-b1b2f91a-d102`) which resolved several limitations.
- When switching workspaces, update three places: `databricks.yml`, `.databricks/.databricks.env`, and `~/.databrickscfg`.
- After switching workspace, delete `.databricks/bundle/dev/terraform/terraform.tfstate` — stale state causes permission errors on the new workspace's jobs.

### Catalog / Schema

- New workspace uses `mlops_dev.dineshka` (catalog.schema). Updated `project_config.yml` for `dev` and `acc` targets.

### Bundle Deployment (`databricks bundle deploy`)

- Databricks CLI v0.250.0 uses Terraform provider v1.75.0 internally.
- This provider requires `client: "1"` in the job environment spec and does **not** recognise `environment_version: "4"` (the field used by the reference project / newer provider).
- Setting only `environment_version: "4"` in the YAML causes a hard Terraform error: `The argument "client" is required`.
- Setting only `client: "1"` deploys successfully but creates a job on an older serverless runtime that **does not support custom library installation**.
- **Workaround**: deploy with `client: "1"` to satisfy Terraform, then immediately patch the job via the raw REST API to set `environment_version: "4"`:
  ```bash
  databricks api post /api/2.1/jobs/update --profile llm-ops-course-dink --json '{
    "job_id": <JOB_ID>,
    "new_settings": {
      "environments": [{
        "environment_key": "default",
        "spec": {
          "environment_version": "4",
          "dependencies": ["<workspace_whl_path>"]
        }
      }]
    }
  }'
  ```
- After every `bundle deploy`, this patch must be re-applied (deploy resets the job to `client: "1"`).

### Wheel / Library Installation

- The `../dist/*.whl` glob in the job YAML is **not resolved** by the bundle for `client: "1"` environments — it gets passed as-is to pip which can't find it. Use the full workspace path instead:
  ```
  /Workspace/Users/<user>/.bundle/<bundle-name>/<target>/artifacts/.internal/<wheel>.whl
  ```
  In YAML: `/Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}/artifacts/.internal/causal_inference_curator-0.0.1-py3-none-any.whl`
- Once `environment_version: "4"` is active, library installation works with the full reference dependency list.

### Python Version

- Reference project and `CLAUDE.md` specify **Python 3.12** (`>=3.12, <3.13`). Stick with this.
- The Free Edition workspace ran Python 3.11.10 which rejected the wheel. The new paid workspace runs Python 3.12 on `environment_version: "4"`.

### PDF Ingestion Notebook (`1.3_causal_inference_data_ingestion.py`)

- PDFs live in a Unity Catalog Volume: `/Volumes/mlops_dev/dineshka/causal_inference_pdfs/`
- Notebook was updated to derive the path dynamically from config instead of a hardcoded `/Workspace/Shared/...` path:
  ```python
  volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{cfg.project.volume}"
  papers = fetch_pdf_metadata(volume_path)
  ```
- Output Delta table: `mlops_dev.dineshka.causal_inference_papers`

### Authentication

- Profile name in `~/.databrickscfg`: `llm-ops-course-dink`
- Always pass `--profile llm-ops-course-dink` to `databricks` CLI commands (two profiles matched the new host, causing ambiguity).

## Week 2: Context Engineering & Vector Search

### What We Built

- **PDF parsing pipeline** (`data_processor.py`): reads causal inference PDFs from the UC Volume, parses them with `ai_parse_document` via SQL, filters out figures/tables, and writes clean text chunks to `causal_inference_chunks` (CDF enabled)
- **Vector search** (`vector_search.py`): creates a delta-sync index on `llmops_course_vs_endpoint` using `databricks-gte-large-en`; `sync_index()` handles incremental updates when new PDFs are added
- **DAB job** (`resources/process_data.yml` + `resources/deployment_scripts/process_data.py`): production pipeline that runs `DataProcessor.process_and_save()` then `sync_index()` — add a new PDF to the volume, run the job, knowledge base is updated
- **Notebooks 2.1–2.4**: interactive exploration of context engineering theory, PDF parsing, chunking strategies, and embeddings/vector search (semantic, filtered, hybrid, reranked)

### Key Decisions

- `ai_parse_document` is called via SQL `INSERT INTO ... SELECT` (not PySpark `F.expr`) — returns a JSON string that we parse with a Python UDF
- `element_type` is used to filter chunks during processing but is **not stored** in the chunks table
- Index type is `TRIGGERED` (not `CONTINUOUS`) — sync must be triggered manually or via the DAB job
- Used the shared `llmops_course_vs_endpoint` (created by course instructor); the index `mlops_dev.dineshka.causal_inference_chunks_index` is ours

### Dependency Fixes

- `numpy==2.4.0` does not exist on PyPI — fixed to `numpy==2.2.6`
- `arxiv==2.3.1` was a leftover from the reference project — removed

### Notebook Magic Cell Format

- Bare `%pip install` is invalid Python syntax — must use `# MAGIC %pip install ...` followed by `# MAGIC %restart_python`
- `%restart_python` is Databricks-only — notebooks with this cell cannot run fully outside Databricks (though the non-magic cells work fine locally via the VS Code extension)

### Serverless v4

- Switched to Serverless Runtime v4 (Python 3.12) in VS Code Databricks extension settings — aligns with `requires-python = ">=3.12"` and resolved library install issues

### Branch Strategy

- `week2` branch contains all week 2 work; PR open for review
- `week3` was branched off `week2` (not `main`) so it includes all week 2 changes and we can build on top without waiting for the PR to merge
- Same approach going forward: branch each week off the previous week's branch

### Week 3 Approach

- Instructor reference notebooks (starting with `3`) will be added to the `llm_ops_reference` folder as usual
- We adapt from the reference to our causal inference use case, same as week 1 and week 2

---

## Week 4: MLflow Tracing & Evaluation

### What We Built

- **`src/causal_inference_curator/agent.py`** — `CausalInferenceAgent(ResponsesAgent)` with full MLflow tracing:
  - `predict_stream()` decorated `@mlflow.trace(AGENT)` — root span
  - `load_memory()` decorated `@mlflow.trace(RETRIEVER)` — Lakebase session load
  - `call_and_run_tools()` decorated `@mlflow.trace(CHAIN)` — orchestration loop
  - `call_llm()` uses `mlflow.start_span(LLM)` manually (streaming-compatible)
  - `execute_tool()` decorated `@mlflow.trace(TOOL)` — vector search calls
  - `save_memory()` decorated `@mlflow.trace(CHAIN)` — Lakebase session save
  - `log_register_agent()` helper to log + register to Unity Catalog
- **`src/causal_inference_curator/evaluation.py`** — Domain-specific scorers:
  - `polite_tone_guideline` — LLM-as-judge, binary
  - `scope_guideline` — stays on causal inference topics
  - `cites_sources_guideline` — claims grounded in retrieved papers
  - `word_count_check` — deterministic, < 400 words
  - `mentions_papers` — deterministic, cites research
  - `uses_causal_terminology` — deterministic, domain language check
  - `evaluate_agent()` — convenience wrapper for full eval run
- **`causal_inference_agent.py`** (repo root) — MLflow pyfunc entry-point used by `log_model(python_model=...)`
- **`eval_inputs.txt`** (repo root) — 10 causal inference evaluation questions
- **Notebooks 4.1–4.4**: tracing basics → traced agent → evaluation theory → log & register

### Config Changes

- Added `lakebase_project_id` field to `ProjectConfig` (optional, for future use)
- Added `experiment_name` field to `ProjectConfig` (default: `/Shared/causal-inference-agent`)
- Added `experiment_path` property as alias for `experiment_name`
- Updated `project_config.yml` for all envs with `lakebase_project_id: null` and `experiment_name`

### Key Differences from Reference

- Reference `ArxivAgent` uses `lakebase_project_id` in `LakebaseMemory(project_id=...)` — our `LakebaseMemory` uses `host` + `instance_name` (kept our interface, agent constructor takes `lakebase_host` + `lakebase_instance_name`)
- Reference `load_config()` returns `ProjectConfig` only; ours returns `Config` (with `.project`, `.model`, etc.) — notebooks use `cfg.project.llm_endpoint` etc.
- Our evaluation scorers are causal-inference-specific: `uses_causal_terminology`, `cites_sources_guideline` instead of the reference's `hook_in_post_guideline`
- Skipped `DatabricksGenieSpace` / `DatabricksSQLWarehouse` in resources (Genie not configured)

### MLflow Registry URI Quirk (Local / Databricks Connect)

- When running locally via Databricks Connect, `mlflow.register_model` tries to read `spark.mlflow.modelRegistryUri` via Spark GRPC first — this always fails with `CONFIG_NOT_AVAILABLE` noise in the logs but is harmless
- The fix is to explicitly call `mlflow.set_registry_uri(f"databricks-uc://{profile}")` **before** any `register_model` call
- Every notebook's setup block must include this, and any helper function that calls `register_model` internally must also set it
- The GRPC error spam in the logs can be ignored — if the final output says "Registered model" or "Model logged", it worked

### Notebook Runability

- **4.1, 4.2, 4.3**: Run locally via Databricks Connect (VS Code extension). MLflow tracking URI is set to `databricks://{profile}` when not on Databricks runtime.
- **4.4**: Must run on Databricks. `mlflow.pyfunc.log_model(python_model=...)` with a file path needs the Databricks runtime to resolve and upload the agent code.

### Branch Strategy

- `week4` branched off `week3` (same pattern as previous weeks)

---

## Week 6: Trace Propagation, Evaluation & Dashboard

### What We Built

- **`notebooks/6.1_propagate_traces.py`** — sends 30 causal inference queries to the live endpoint to generate MLflow traces
- **`resources/deployment_scripts/update_traces_aggregated.py`** — fetches unevaluated traces, runs `word_count_check` on all, samples 10% for `polite_tone` + `cites_sources` LLM-judge eval, creates/refreshes `causal_inference_traces_aggregated` SQL view
- **`resources/update_traces_aggregated.yml`** — DAB job for the pipeline above
- **`resources/dashboards/agent_monitoring_dashboard.yml` + `.lvdash.json`** — Lakeview dashboard reading from the aggregated view; shows latency, token usage, quality pass rates over time

### Traces Table Name

- Inference table: `mlops_dev.dineshka.trace_logs_2065492370656966` (the suffix is the MLflow experiment ID)
- Found by listing `SHOW TABLES IN mlops_dev.dineshka` after running the endpoint

### Token Acquisition (Local)

- `workspace.tokens.create()` is blocked in this workspace — use `workspace.config.authenticate()["Authorization"].split(" ")[1]` instead
- This works both locally (VS Code metadata-service) and on Databricks runtime

### LLM-Judge Feedback: None Guard

- `mlflow.genai.evaluate` with `Guidelines` scorers can return `None` for `feedback.value` if the judge can't assess the response
- `mlflow.log_feedback(value=None)` raises `MlflowException` — guard with `if val is None: continue`

### Dashboard deploy: pre-create `resources/` workspace dir

- Terraform provider v1.75.0 (`databricks_dashboard`) requires the parent workspace path to exist before creating the dashboard
- `bundle deploy` does **not** auto-create it — pre-create manually once per fresh target:
  ```python
  from databricks.sdk import WorkspaceClient
  w = WorkspaceClient()
  w.workspace.mkdirs('/Workspace/Users/<user>/.bundle/<bundle>/<target>/resources')
  ```
- After that, subsequent `bundle deploy` runs work without this step

### Dashboard `file_path` Gotcha

- `file_path` in the dashboard YAML is relative to the bundle root, **not** to the YAML file location
- Use `../dashboards/agent_monitoring_dashboard.lvdash.json` (relative from `resources/` to sibling `resources/dashboards/`)

### Warehouse ID

- Dashboard needs a SQL warehouse: `7077f4cb0e616c62` (shared course warehouse, same workspace)
- Set in both `project_config.yml` (all envs) and as `databricks.yml` variable default

### Running Locally

- `6.1_propagate_traces.py`: runs directly with `uv run python` + `DATABRICKS_CONFIG_PROFILE`
- `update_traces_aggregated.py`: source `.databricks/.databricks.env` first (sets `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`), then run from `resources/deployment_scripts/` so `../../project_config.yml` resolves correctly

### Branch Strategy

- `week6` branched from `main` after week 5 merged

---

### Git / SSH

- `~/.ssh/config` routes all `github.com` connections to the Adobe key (`id_ed25519_orgb`) via `IdentitiesOnly yes`. This breaks pushes/pulls for the personal llmops org.
- **Do NOT modify `~/.ssh/config`** — it affects Adobe work.
- Instead, always use `GIT_SSH_COMMAND` to specify the correct key for this repo:
  ```bash
  GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" git push
  GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" git pull origin main
  ```
- This is scoped to a single command and doesn't affect any other sessions or repos.
- The correct key fingerprint for this repo: `SHA256:Hg413ND3POpYOg1EacFp1uPeXN2NkwBPtDUmfCZ4Qos` (`~/.ssh/id_ed25519`).
