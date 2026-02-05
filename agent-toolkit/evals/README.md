# Evaluation Framework

Evaluate research quality across four dimensions using LLM-as-judge methodology.

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Grounding** | Claim-to-source attribution | Trust diagnostics, hallucination detection |
| **Content Attribution** | Web vs internal vs prior knowledge | Quantify value of web data |
| **Relevance** | Source-query alignment | Assess retrieval quality |
| **Search Quality** | Query effectiveness, coverage | Optimize search strategies |
| **Custom Dataset** | Model accuracy on query-answer pairs | Benchmark RAG pipelines, regression testing |

## Quick Start

```python
from tavily_agent_toolkit import ModelConfig, ModelObject
from tavily_agent_toolkit.evals import evaluate_research

# Evaluate a research output
result = await evaluate_research(
    query="What are the latest AI trends?",
    report=generated_report,
    web_sources=sources,
    judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
    metrics=["grounding", "relevance"],  # Optional: defaults to all
)

print(f"Grounding: {result.grounding.grounding_ratio:.1%}")
print(f"Relevance: {result.relevance.source_relevance_score:.2f}")
print(f"Overall: {result.overall_score:.2f}")
```

## Metrics

### Grounding
Measures how well report claims are supported by sources.

```python
from tavily_agent_toolkit.evals import compute_grounding_metrics

result = await compute_grounding_metrics(
    report="NVIDIA reported $26B revenue...",
    sources=[{"url": "...", "content": "..."}],
    judge_model_config=config,
)

# Key metrics:
# - grounding_ratio: % of claims supported by sources
# - citation_accuracy: % of citations that are accurate
# - unsupported_claims_count: Number of unsupported claims
```

### Content Attribution
Breaks down content by source type to quantify web data value.

```python
from tavily_agent_toolkit.evals import compute_content_attribution_metrics

result = await compute_content_attribution_metrics(
    report=report,
    web_sources=sources,
    judge_model_config=config,
)

# Key metrics:
# - web_content_ratio: % from web sources
# - prior_knowledge_ratio: % from model's training data
# - source_diversity: Unique domains / total sources
```

### Relevance
Evaluates how well sources address the query.

```python
from tavily_agent_toolkit.evals import compute_relevance_metrics

result = await compute_relevance_metrics(
    query="What are AI trends?",
    sources=search_results,
    judge_model_config=config,
)

# Key metrics:
# - source_relevance_score: Average relevance (0-1)
# - answer_coverage: % of query aspects covered
# - top_k_precision: % of top-k sources that are relevant
```

### Search Quality
Evaluates query generation and result coverage.

```python
from tavily_agent_toolkit.evals import compute_search_quality_metrics

result = await compute_search_quality_metrics(
    research_task="Analyze AI market",
    queries=["AI market size", "AI trends 2024"],
    results=search_results,
    judge_model_config=config,
    credits_used=10,
)

# Key metrics:
# - query_quality_score: Generated query effectiveness
# - result_coverage: Topic coverage
# - deduplication_efficiency: Unique content ratio
# - credit_efficiency: Quality per credit
```

### Custom Dataset Evaluation
Evaluate your model against custom query-answer datasets.

```python
from tavily_agent_toolkit.evals import evaluate_dataset, CSVDataset

# Load your dataset (CSV with query, expected_answer columns)
dataset = CSVDataset("my_queries.csv")

# Define your model as an async callable
async def my_model(query: str) -> str:
    # Your RAG/search/LLM logic here
    return "answer"

# Run evaluation
result = await evaluate_dataset(
    dataset=dataset,
    model=my_model,
    judge_model_config=config,
)

# Key metrics:
# - accuracy: % of correct answers
# - by_category: accuracy breakdown by category
# - by_difficulty: accuracy breakdown by difficulty level
print(f"Accuracy: {result.accuracy:.1%}")
```

## Integration with Tools

```python
from tavily_agent_toolkit import search_and_answer
from tavily_agent_toolkit.evals import evaluate_research

# Run research
result = await search_and_answer(query=query, ...)

# Self-evaluate
eval_result = await evaluate_research(
    query=query,
    report=result["answer"],
    web_sources=result["results"],
    judge_model_config=config,
)

# Attach metrics to result
result["quality_metrics"] = eval_result.to_dict()
```

## Tutorials

| Notebook | Description |
|----------|-------------|
| [grounding-eval.ipynb](./grounding-eval.ipynb) | Evaluating claim-to-source attribution and detecting hallucinations |
| [content-attribution.ipynb](./content-attribution.ipynb) | Measuring web vs internal vs prior knowledge breakdown |
| [search-quality.ipynb](./search-quality.ipynb) | Evaluating query effectiveness and result quality |
| [custom-dataset-eval.ipynb](./custom-dataset-eval.ipynb) | Testing models against custom query-answer datasets |

## Prerequisites

```bash
pip install tavily-agent-toolkit
```

Set environment variables:
```bash
export TAVILY_API_KEY=your_key
export OPENAI_API_KEY=your_key  # or other LLM provider
```

## Architecture

```
evals/
├── models.py          # Data models (EvalResult, EvalUsage, DatasetEvalResult, etc.)
├── datasets/          # Dataset loaders
│   ├── base.py        # DatasetItem, BaseDataset, InMemoryDataset
│   └── csv_dataset.py # CSV file loader
├── metrics/           # Metric computation
│   ├── grounding.py
│   ├── relevance.py
│   ├── content_attribution.py
│   ├── search_quality.py
│   └── correctness.py # Dataset correctness metrics
├── judges/            # LLM-as-judge implementations
│   ├── base.py
│   ├── grounding_judge.py
│   ├── relevance_judge.py
│   ├── quality_judge.py
│   └── correctness_judge.py # CORRECT/INCORRECT grading
├── evaluators/        # High-level orchestrators
│   ├── research_evaluator.py
│   ├── retrieval_evaluator.py
│   └── dataset_evaluator.py # Custom dataset evaluation
├── grounding-eval.ipynb
├── content-attribution.ipynb
├── search-quality.ipynb
└── custom-dataset-eval.ipynb
```
