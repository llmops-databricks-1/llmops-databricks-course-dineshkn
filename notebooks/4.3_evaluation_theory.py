# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.3: GenAI Evaluation Theory
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Why evaluation matters for GenAI
# MAGIC - Types of evaluation metrics
# MAGIC - MLflow evaluation framework
# MAGIC - Guidelines vs Judges
# MAGIC - Custom code-based scorers
# MAGIC - Combining multiple scorers
# MAGIC - Judge alignment with human feedback

# COMMAND ----------

import os
from typing import Literal

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow.genai.judges import make_judge
from pyspark.sql import SparkSession

from causal_inference_curator.config import get_env, load_config
from causal_inference_curator.evaluation import (
    cites_sources_guideline,
    mentions_papers,
    polite_tone_guideline,
    scope_guideline,
    uses_causal_terminology,
    word_count_check,
)

# COMMAND ----------

# Setup
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ.get("PROFILE", "llm-ops-course-dink")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

mlflow.set_experiment(cfg.project.experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Why Evaluation Matters for GenAI
# MAGIC
# MAGIC Traditional ML evaluation (accuracy, F1, etc.) doesn't work well for GenAI because:
# MAGIC
# MAGIC ### Challenges:
# MAGIC - **Open-ended outputs**: No single "correct" answer to "explain DiD"
# MAGIC - **Subjective quality**: What's "good" varies by audience (student vs researcher)
# MAGIC - **Multiple dimensions**: Accuracy, tone, citation quality, scope, length
# MAGIC - **Context-dependent**: A brief answer is fine for a definition; wrong for a methodology question
# MAGIC
# MAGIC ### Why Evaluate?
# MAGIC 1. **Quality assurance**: Ensure responses meet research standards
# MAGIC 2. **Regression detection**: Catch degradation when you change the model or prompt
# MAGIC 3. **Model comparison**: Choose the best LLM for causal inference questions
# MAGIC 4. **Continuous improvement**: Identify where the agent fails (wrong papers? bad citations?)
# MAGIC 5. **Trust**: Demonstrate the agent is reliable to end users

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Types of Evaluation Metrics
# MAGIC
# MAGIC ### A. Automated / Code-Based Metrics (deterministic, cheap)
# MAGIC | Metric | What it Measures |
# MAGIC |--------|-----------------|
# MAGIC | `word_count_check` | Response length — too long = hard to read |
# MAGIC | `mentions_papers` | Does the response cite research? |
# MAGIC | `uses_causal_terminology` | Is domain language used correctly? |
# MAGIC
# MAGIC ### B. LLM-as-Judge Metrics (nuanced, more expensive)
# MAGIC | Metric | What it Measures |
# MAGIC |--------|-----------------|
# MAGIC | `polite_tone_guideline` | Professional, non-dismissive tone |
# MAGIC | `scope_guideline` | Stays on causal inference topics |
# MAGIC | `cites_sources_guideline` | Claims are grounded in retrieved papers |
# MAGIC | `quality_judge` | Overall response quality (1–5) |
# MAGIC
# MAGIC ### C. Human Evaluation
# MAGIC - Most reliable but expensive and slow
# MAGIC - Used to calibrate and validate automated judges

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Guidelines — Binary Pass/Fail (from our evaluation.py)

# COMMAND ----------

logger.info("Guidelines loaded from causal_inference_curator.evaluation:")
logger.info(f"  polite_tone_guideline  — {len(polite_tone_guideline.guidelines)} rules")
logger.info(f"  scope_guideline        — {len(scope_guideline.guidelines)} rules")
logger.info(
    f"  cites_sources_guideline — {len(cites_sources_guideline.guidelines)} rules"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Guidelines on Sample Responses

# COMMAND ----------

test_data = [
    {
        "inputs": {"question": "What is the difference-in-differences method?"},
        "outputs": "Just read a textbook, it's basic stuff.",
    },
    {
        "inputs": {"question": "What is the difference-in-differences method?"},
        "outputs": (
            "I'd be happy to help! Difference-in-differences (DiD) is a quasi-experimental "
            "technique that compares the change in outcomes over time between a treatment group "
            "and a control group. It relies on the parallel trends assumption."
        ),
    },
]

results = mlflow.genai.evaluate(data=test_data, scorers=[polite_tone_guideline])

logger.info("Polite Tone Guideline Results:")
logger.info("=" * 80)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Judges — Scored Evaluation (numeric)

# COMMAND ----------

quality_judge = make_judge(
    name="response_quality",
    instructions=(
        "Evaluate the quality of the response in {{ outputs }} to the causal inference "
        "question in {{ inputs }}. Score from 1 to 5:\n"
        "1 - Completely unhelpful, incorrect, or off-topic\n"
        "2 - Partially helpful but missing key concepts\n"
        "3 - Adequate — covers the basics but lacks depth\n"
        "4 - Good — clear, accurate, and well-explained\n"
        "5 - Excellent — comprehensive, cites relevant methods/papers, "
        "technically precise"
    ),
    model=f"databricks:/{cfg.project.llm_endpoint}",
    feedback_value_type=int,
)

logger.info(f"Quality Judge created — scores 1-5 using {cfg.project.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Judge

# COMMAND ----------

judge_test_data = [
    {
        "inputs": {"question": "What is an instrumental variable?"},
        "outputs": "It's a variable that helps with causality.",
    },
    {
        "inputs": {"question": "What is an instrumental variable?"},
        "outputs": (
            "An instrumental variable (IV) is a variable that is correlated with the "
            "endogenous explanatory variable but uncorrelated with the error term. "
            "It satisfies two conditions: relevance (correlated with the treatment) "
            "and exclusion restriction (affects the outcome only through the treatment). "
            "IVs are used to obtain consistent estimates when OLS would be biased due "
            "to endogeneity."
        ),
    },
]

judge_results = mlflow.genai.evaluate(data=judge_test_data, scorers=[quality_judge])

logger.info("Quality Judge Results:")
logger.info("=" * 80)
display(judge_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Custom Code-Based Scorers (from our evaluation.py)

# COMMAND ----------

logger.info("Custom scorers from causal_inference_curator.evaluation:")
logger.info("  word_count_check        — True if response < 400 words")
logger.info("  mentions_papers         — True if response cites research")
logger.info("  uses_causal_terminology — True if domain terms are present")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Custom Scorers

# COMMAND ----------

custom_test_data = [
    {
        "inputs": {"question": "Explain propensity score matching."},
        "outputs": (
            "Propensity score matching (PSM) is a statistical technique used to estimate "
            "the causal effect of a treatment by accounting for confounding variables. "
            "The propensity score is the probability of receiving treatment given observed "
            "covariates. By matching treated and control units with similar propensity scores, "
            "we approximate a randomized experiment. Rosenbaum and Rubin (1983) introduced "
            "this method in their seminal paper on the central role of the propensity score."
        ),
    },
    {
        "inputs": {"question": "Explain propensity score matching."},
        "outputs": "It's a matching method. " * 80,  # very long, no citations
    },
]

custom_results = mlflow.genai.evaluate(
    data=custom_test_data,
    scorers=[word_count_check, mentions_papers, uses_causal_terminology],
)

logger.info("Custom Scorer Results:")
logger.info("=" * 80)
display(custom_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Categorical Judge

# COMMAND ----------

confidence_judge = make_judge(
    name="response_confidence",
    instructions=(
        "Analyse the confidence level expressed in the response in {{ outputs }}. "
        "Classify as: 'confident', 'hedged', or 'uncertain'"
    ),
    feedback_value_type=Literal["confident", "hedged", "uncertain"],
    model=f"databricks:/{cfg.project.llm_endpoint}",
)

logger.info("Categorical Judge created — classifies confidence level")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comprehensive Evaluation — All Scorers Combined

# COMMAND ----------

all_scorers = [
    polite_tone_guideline,
    cites_sources_guideline,
    quality_judge,
    word_count_check,
    mentions_papers,
    uses_causal_terminology,
    confidence_judge,
]

comprehensive_test_data = [
    {
        "inputs": {"question": "What is the regression discontinuity design?"},
        "outputs": (
            "Regression discontinuity design (RDD) is a quasi-experimental approach that "
            "exploits a known threshold (cutoff) in an assignment variable to estimate "
            "causal effects. Units just above and below the cutoff are assumed to be "
            "comparable, so differences in outcomes at the cutoff can be attributed to "
            "the treatment. Thistlethwaite and Campbell (1960) first introduced RDD, and "
            "it has since become a standard tool in program evaluation."
        ),
    },
]

comprehensive_results = mlflow.genai.evaluate(
    data=comprehensive_test_data,
    scorers=all_scorers,
)

logger.info("Comprehensive Evaluation Results:")
logger.info("=" * 80)
display(comprehensive_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Judge Alignment with Human Feedback
# MAGIC
# MAGIC ### The Problem:
# MAGIC - LLM judges may not align with what a causal inference expert considers "good"
# MAGIC - A judge trained on general text may not understand domain-specific quality
# MAGIC
# MAGIC ### The Solution: SIMBA Alignment
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer
# MAGIC
# MAGIC # Collect traces where a domain expert has provided feedback
# MAGIC traces_with_feedback = get_traces_with_human_feedback()
# MAGIC
# MAGIC # Align the judge to match expert preferences
# MAGIC optimizer = SIMBAAlignmentOptimizer(model="databricks:/my-llm")
# MAGIC aligned_judge = quality_judge.align(optimizer, traces_with_feedback)
# MAGIC ```
# MAGIC
# MAGIC ### Workflow:
# MAGIC 1. Run the agent on test questions
# MAGIC 2. Have a causal inference expert rate the responses
# MAGIC 3. Compare judge scores vs expert scores
# MAGIC 4. Use SIMBA to optimise judge instructions to match expert preferences
# MAGIC 5. Use the aligned judge for production evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Best Practices
# MAGIC
# MAGIC ### ✅ Do:
# MAGIC 1. **Use multiple scorers** — no single metric captures everything
# MAGIC 2. **Mix automated + LLM judges** — cheap checks first, expensive judges second
# MAGIC 3. **Create domain-specific guidelines** — generic "polite tone" isn't enough
# MAGIC 4. **Validate judges with domain experts** periodically
# MAGIC 5. **Track metrics over time** — detect regressions when you update the model
# MAGIC 6. **Test edge cases** — ambiguous questions, off-topic queries, long questions
# MAGIC 7. **Version your eval sets** — so you can compare across model versions
# MAGIC
# MAGIC ### ❌ Don't:
# MAGIC 1. Rely on a single metric
# MAGIC 2. Use the same model as both generator and judge (self-evaluation bias)
# MAGIC 3. Evaluate on too few examples (< 10 is usually not meaningful)
# MAGIC 4. Forget to log evaluation results alongside the model in MLflow
