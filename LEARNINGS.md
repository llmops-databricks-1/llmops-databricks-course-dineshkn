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
