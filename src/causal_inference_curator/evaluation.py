"""Evaluation scorers and guidelines for the Causal Inference Agent."""

import mlflow
from mlflow.genai.scorers import Guidelines

from causal_inference_curator.agent import CausalInferenceAgent
from causal_inference_curator.config import ProjectConfig

# ---------------------------------------------------------------------------
# Guidelines — binary pass/fail evaluated by an LLM judge
# ---------------------------------------------------------------------------

polite_tone_guideline = Guidelines(
    name="polite_tone",
    guidelines=[
        "The response must use a polite and professional tone throughout",
        "The response should be friendly and helpful without being condescending",
        "The response must avoid any dismissive or rude language",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

scope_guideline = Guidelines(
    name="stays_in_scope",
    guidelines=[
        "The response must only discuss topics related to causal inference, "
        "econometrics, statistics, or research methodology",
        "The response should not answer questions about completely unrelated topics",
        "If asked about non-causal-inference topics, politely redirect to "
        "causal inference research questions",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

cites_sources_guideline = Guidelines(
    name="cites_sources",
    guidelines=[
        "The response should reference specific papers, authors, or studies "
        "when making factual claims about research findings",
        "The response must not present information as fact without grounding it "
        "in the retrieved documents",
        "Vague assertions like 'research shows' without citing a source are "
        "not acceptable when papers are available",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)


# ---------------------------------------------------------------------------
# Custom code-based scorers — deterministic, no LLM call needed
# ---------------------------------------------------------------------------


@mlflow.genai.scorer
def word_count_check(outputs: list) -> bool:
    """Return True if the response is under 400 words.

    Causal inference explanations can be technical; we allow slightly more
    words than the reference (350) to accommodate necessary detail.
    """
    text = _extract_text(outputs)
    return len(text.split()) < 400


@mlflow.genai.scorer
def mentions_papers(outputs: list) -> bool:
    """Return True if the response references specific research papers or studies."""
    text = _extract_text(outputs).lower()
    keywords = [
        "paper",
        "study",
        "research",
        "author",
        "published",
        "journal",
        "et al",
        "findings",
        "experiment",
        "dataset",
    ]
    return any(kw in text for kw in keywords)


@mlflow.genai.scorer
def uses_causal_terminology(outputs: list) -> bool:
    """Return True if the response uses domain-appropriate causal inference language."""
    text = _extract_text(outputs).lower()
    causal_terms = [
        "causal",
        "counterfactual",
        "treatment",
        "control",
        "confounder",
        "instrumental variable",
        "regression discontinuity",
        "difference-in-differences",
        "propensity score",
        "randomized",
        "potential outcome",
        "ate",
        "att",
        "selection bias",
        "endogeneity",
        "identification",
    ]
    return any(term in text for term in causal_terms)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_text(outputs: object) -> str:
    """Normalise the outputs argument to a plain string."""
    if isinstance(outputs, list) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, dict):
            return first.get("text", str(first))
        return str(first)
    return str(outputs)


# ---------------------------------------------------------------------------
# Convenience: run a full evaluation against the agent
# ---------------------------------------------------------------------------


def evaluate_agent(
    cfg: ProjectConfig,
    eval_inputs_path: str,
) -> "mlflow.models.EvaluationResult":
    """Run evaluation on the CausalInferenceAgent.

    Args:
        cfg: Project configuration.
        eval_inputs_path: Path to a plain-text file with one question per line.

    Returns:
        MLflow EvaluationResult with aggregated metrics.
    """
    agent = CausalInferenceAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=cfg.system_prompt,
        catalog=cfg.catalog,
        schema=cfg.schema,
    )

    with open(eval_inputs_path) as f:
        eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]

    def predict_fn(question: str) -> str:
        request = {"input": [{"role": "user", "content": question}]}
        result = agent.predict(request)
        return result.output[-1].content

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[
            word_count_check,
            polite_tone_guideline,
            cites_sources_guideline,
            mentions_papers,
            uses_causal_terminology,
        ],
    )


def create_eval_data_from_file(eval_inputs_path: str) -> list[dict]:
    """Load evaluation data from a plain-text file (one question per line).

    Args:
        eval_inputs_path: Path to the evaluation inputs file.

    Returns:
        List of dicts in the format expected by mlflow.genai.evaluate.
    """
    with open(eval_inputs_path) as f:
        return [{"inputs": {"question": line.strip()}} for line in f if line.strip()]
