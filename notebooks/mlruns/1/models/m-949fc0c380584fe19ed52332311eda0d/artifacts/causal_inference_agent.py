"""MLflow pyfunc entry-point for the CausalInferenceAgent.

This file is referenced by mlflow.pyfunc.log_model(python_model=...) in
notebook 4.4. It must live at the repo root so that the relative path
'../causal_inference_agent.py' resolves correctly from the notebooks/ folder.

When loaded by MLflow serving, the module-level code instantiates the agent
from the model_config injected at log time.
"""

import mlflow

from causal_inference_curator.agent import CausalInferenceAgent
from causal_inference_curator.config import ProjectConfig


def _load_agent() -> CausalInferenceAgent:
    """Instantiate the agent from the MLflow model config."""
    model_config = mlflow.models.ModelConfig(development_config="../project_config.yml")

    cfg = ProjectConfig(
        catalog=model_config.get("catalog"),
        schema=model_config.get("schema"),
        volume="causal_inference_pdfs",
        llm_endpoint=model_config.get("llm_endpoint"),
        embedding_endpoint="databricks-gte-large-en",
        warehouse_id="",
        vector_search_endpoint="llmops_course_vs_endpoint",
        system_prompt=model_config.get("system_prompt"),
    )

    return CausalInferenceAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=cfg.system_prompt,
        catalog=cfg.catalog,
        schema=cfg.schema,
    )


agent = _load_agent()
mlflow.models.set_model(agent)
