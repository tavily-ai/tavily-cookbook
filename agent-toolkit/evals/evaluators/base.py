"""Base evaluator class for orchestrating evaluations."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from ..models import EvalResult, BatchEvalResult, EvalUsage

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


MetricType = Literal["relevance", "grounding", "content_attribution", "search_quality"]


class BaseEvaluator(ABC):
    """Abstract base class for evaluation orchestrators.

    Evaluators coordinate multiple metrics and judges to produce
    comprehensive evaluation results.

    Subclasses should implement:
    - evaluate(): Evaluate a single item
    - evaluate_batch(): Evaluate multiple items

    Example:
        class MyEvaluator(BaseEvaluator):
            async def evaluate(self, **kwargs) -> EvalResult:
                # Run relevant metrics
                result = EvalResult(query=kwargs["query"])
                if "grounding" in self.metrics:
                    result.grounding = await compute_grounding_metrics(...)
                return result
    """

    def __init__(
        self,
        judge_model_config: "ModelConfig",
        metrics: Optional[list[MetricType]] = None,
    ):
        """Initialize the evaluator.

        Args:
            judge_model_config: ModelConfig for judge LLM calls
            metrics: List of metrics to compute. Defaults to all available.
        """
        self.judge_model_config = judge_model_config
        self.metrics = metrics or ["relevance", "grounding", "content_attribution", "search_quality"]
        self.usage = EvalUsage()

    def reset_usage(self) -> None:
        """Reset accumulated usage statistics."""
        self.usage = EvalUsage()

    @abstractmethod
    async def evaluate(self, **kwargs: Any) -> EvalResult:
        """Evaluate a single item.

        Subclasses must implement this method.

        Returns:
            EvalResult with computed metrics
        """
        pass

    async def evaluate_batch(
        self,
        items: list[dict],
        parallel: bool = False,
    ) -> BatchEvalResult:
        """Evaluate a batch of items.

        Args:
            items: List of dicts with evaluation inputs
            parallel: Whether to run evaluations in parallel

        Returns:
            BatchEvalResult with aggregated results
        """
        import asyncio

        self.reset_usage()
        results = []

        if parallel:
            # Run all evaluations concurrently
            tasks = [self.evaluate(**item) for item in items]
            results = await asyncio.gather(*tasks)
        else:
            # Run sequentially
            for item in items:
                result = await self.evaluate(**item)
                results.append(result)

        # Create batch result
        batch_result = BatchEvalResult(results=results)
        batch_result.compute_aggregates()

        return batch_result

    def get_usage(self) -> EvalUsage:
        """Get accumulated usage statistics.

        Returns:
            Copy of current EvalUsage
        """
        return EvalUsage(
            judge_llm_calls=self.usage.judge_llm_calls,
            judge_input_tokens=self.usage.judge_input_tokens,
            judge_output_tokens=self.usage.judge_output_tokens,
            eval_response_time=self.usage.eval_response_time,
        )
