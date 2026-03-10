"""Dataset evaluator for custom query-answer pair evaluation."""

import asyncio
from typing import Any, Awaitable, Callable, Optional

from ..models import (
    DatasetEvalResult,
    ItemResult,
    CategoryBreakdown,
    EvalUsage,
)
from ..datasets.base import BaseDataset, DatasetItem
from ..judges.correctness_judge import CorrectnessJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


# Type alias for model callable
ModelCallable = Callable[[str], Awaitable[str]]


class DatasetEvaluator:
    """Evaluates a model against a dataset of query-answer pairs.

    This evaluator orchestrates the evaluation workflow:
    1. Iterates through dataset items
    2. Calls the model for each query
    3. Uses an LLM judge to grade each prediction
    4. Aggregates results with category/difficulty breakdowns

    Example:
        from tavily_agent_toolkit import ModelConfig, ModelObject
        from tavily_agent_toolkit.evals import DatasetEvaluator, CSVDataset

        # Load dataset
        dataset = CSVDataset("my_queries.csv")

        # Define your model
        async def my_model(query: str) -> str:
            # Your model logic here
            return "answer"

        # Create evaluator
        evaluator = DatasetEvaluator(
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )

        # Run evaluation
        result = await evaluator.evaluate(dataset, my_model)
        print(f"Accuracy: {result.accuracy:.1%}")
    """

    def __init__(
        self,
        judge_model_config: "ModelConfig",
        grader_prompt: Optional[str] = None,
        parallel: bool = True,
        max_concurrency: int = 10,
    ):
        """Initialize the dataset evaluator.

        Args:
            judge_model_config: ModelConfig for the judge LLM
            grader_prompt: Optional custom prompt for the correctness grader
            parallel: Whether to run model calls and judgments in parallel
            max_concurrency: Maximum concurrent operations when parallel=True
        """
        self.judge_model_config = judge_model_config
        self.grader_prompt = grader_prompt
        self.parallel = parallel
        self.max_concurrency = max_concurrency

    async def evaluate(
        self,
        dataset: BaseDataset,
        model: ModelCallable,
        progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
    ) -> DatasetEvalResult:
        """Evaluate a model against a dataset.

        Args:
            dataset: The dataset to evaluate against
            model: Async callable that takes a query string and returns an answer
            progress_callback: Optional callback(current, total, status) for progress

        Returns:
            DatasetEvalResult with accuracy metrics and per-item results

        Example:
            async def my_model(query: str) -> str:
                # Search and generate answer
                results = await tavily.search(query)
                context = "\\n".join([r["content"] for r in results["results"]])
                response = await openai.chat.completions.create(...)
                return response.choices[0].message.content

            result = await evaluator.evaluate(dataset, my_model)
        """
        items = list(dataset)
        total = len(items)

        if total == 0:
            return DatasetEvalResult(
                accuracy=1.0,
                correct_count=0,
                incorrect_count=0,
                total_count=0,
            )

        # Step 1: Get predictions from the model
        if progress_callback:
            progress_callback(0, total, "Getting predictions...")

        predictions = await self._get_predictions(items, model, progress_callback, total)

        # Step 2: Judge all predictions
        if progress_callback:
            progress_callback(total, total, "Judging predictions...")

        item_results, usage = await self._judge_predictions(items, predictions)

        # Step 3: Compute metrics
        return self._compute_results(item_results, usage)

    async def evaluate_precomputed(
        self,
        items: list[tuple[str, str, str]],
        categories: Optional[list[Optional[str]]] = None,
        difficulties: Optional[list[Optional[str]]] = None,
    ) -> DatasetEvalResult:
        """Evaluate precomputed predictions (no model calls needed).

        Use this when you already have predictions from your model and just
        need to run the evaluation.

        Args:
            items: List of (query, expected_answer, predicted_answer) tuples
            categories: Optional list of categories (same length as items)
            difficulties: Optional list of difficulties (same length as items)

        Returns:
            DatasetEvalResult with accuracy metrics

        Example:
            # If you've already run your model:
            items = [
                ("What is 2+2?", "4", "The answer is 4"),
                ("Capital of France?", "Paris", "London"),
            ]
            result = await evaluator.evaluate_precomputed(items)
        """
        if not items:
            return DatasetEvalResult(
                accuracy=1.0,
                correct_count=0,
                incorrect_count=0,
                total_count=0,
            )

        # Create DatasetItems from tuples
        dataset_items = []
        for i, (query, expected, _) in enumerate(items):
            item = DatasetItem(
                query=query,
                expected_answer=expected,
                category=categories[i] if categories and i < len(categories) else None,
                difficulty=difficulties[i] if difficulties and i < len(difficulties) else None,
            )
            dataset_items.append(item)

        predictions = [p for _, _, p in items]

        # Judge predictions
        item_results, usage = await self._judge_predictions(dataset_items, predictions)

        return self._compute_results(item_results, usage)

    async def _get_predictions(
        self,
        items: list[DatasetItem],
        model: ModelCallable,
        progress_callback: Optional[Callable[[int, int, Optional[str]], None]],
        total: int,
    ) -> list[str]:
        """Get model predictions for all items."""
        predictions: list[str] = []

        if self.parallel:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            completed = [0]  # Use list for closure mutation

            async def predict_with_semaphore(idx: int, item: DatasetItem) -> tuple[int, str]:
                async with semaphore:
                    try:
                        result = await model(item.query)
                    except Exception as e:
                        # If model fails, record the error as the prediction
                        result = f"[ERROR: {type(e).__name__}: {e}]"

                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0], total, None)
                    return idx, result

            tasks = [predict_with_semaphore(i, item) for i, item in enumerate(items)]
            indexed_results = await asyncio.gather(*tasks)
            indexed_results = sorted(indexed_results, key=lambda x: x[0])
            predictions = [r for _, r in indexed_results]
        else:
            for i, item in enumerate(items):
                try:
                    result = await model(item.query)
                except Exception as e:
                    result = f"[ERROR: {type(e).__name__}: {e}]"
                predictions.append(result)
                if progress_callback:
                    progress_callback(i + 1, total, None)

        return predictions

    async def _judge_predictions(
        self,
        items: list[DatasetItem],
        predictions: list[str],
    ) -> tuple[list[ItemResult], EvalUsage]:
        """Judge all predictions using the correctness judge."""
        judge = CorrectnessJudge(
            model_config=self.judge_model_config,
            grader_prompt=self.grader_prompt,
        )

        item_results: list[ItemResult] = []

        if self.parallel:
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def judge_with_semaphore(
                idx: int, item: DatasetItem, predicted: str
            ) -> tuple[int, ItemResult]:
                async with semaphore:
                    judgment = await judge.judge(
                        query=item.query,
                        expected_answer=item.expected_answer,
                        predicted_answer=predicted,
                    )
                    result = ItemResult(
                        query=item.query,
                        expected_answer=item.expected_answer,
                        predicted_answer=predicted,
                        grade=judgment["grade"],
                        score=judgment["score"],
                        reasoning=judgment.get("reasoning"),
                        category=item.category,
                        difficulty=item.difficulty,
                        metadata=item.metadata,
                    )
                    return idx, result

            tasks = [
                judge_with_semaphore(i, item, pred)
                for i, (item, pred) in enumerate(zip(items, predictions))
            ]
            indexed_results = await asyncio.gather(*tasks)
            indexed_results = sorted(indexed_results, key=lambda x: x[0])
            item_results = [r for _, r in indexed_results]
        else:
            for item, predicted in zip(items, predictions):
                judgment = await judge.judge(
                    query=item.query,
                    expected_answer=item.expected_answer,
                    predicted_answer=predicted,
                )
                result = ItemResult(
                    query=item.query,
                    expected_answer=item.expected_answer,
                    predicted_answer=predicted,
                    grade=judgment["grade"],
                    score=judgment["score"],
                    reasoning=judgment.get("reasoning"),
                    category=item.category,
                    difficulty=item.difficulty,
                    metadata=item.metadata,
                )
                item_results.append(result)

        return item_results, judge.get_usage()

    def _compute_results(
        self,
        item_results: list[ItemResult],
        usage: EvalUsage,
    ) -> DatasetEvalResult:
        """Compute aggregate metrics from item results."""
        total = len(item_results)
        correct = sum(1 for r in item_results if r.grade == "CORRECT")
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 1.0

        # Compute category breakdown
        by_category: dict[str, CategoryBreakdown] = {}
        category_results: dict[str, list[ItemResult]] = {}
        for r in item_results:
            if r.category is not None:
                if r.category not in category_results:
                    category_results[r.category] = []
                category_results[r.category].append(r)

        for cat, results in category_results.items():
            cat_total = len(results)
            cat_correct = sum(1 for r in results if r.grade == "CORRECT")
            by_category[cat] = CategoryBreakdown(
                category=cat,
                accuracy=cat_correct / cat_total if cat_total > 0 else 1.0,
                correct_count=cat_correct,
                incorrect_count=cat_total - cat_correct,
                total_count=cat_total,
            )

        # Compute difficulty breakdown
        by_difficulty: dict[str, CategoryBreakdown] = {}
        difficulty_results: dict[str, list[ItemResult]] = {}
        for r in item_results:
            if r.difficulty is not None:
                if r.difficulty not in difficulty_results:
                    difficulty_results[r.difficulty] = []
                difficulty_results[r.difficulty].append(r)

        for diff, results in difficulty_results.items():
            diff_total = len(results)
            diff_correct = sum(1 for r in results if r.grade == "CORRECT")
            by_difficulty[diff] = CategoryBreakdown(
                category=diff,  # Reusing CategoryBreakdown structure
                accuracy=diff_correct / diff_total if diff_total > 0 else 1.0,
                correct_count=diff_correct,
                incorrect_count=diff_total - diff_correct,
                total_count=diff_total,
            )

        return DatasetEvalResult(
            accuracy=accuracy,
            correct_count=correct,
            incorrect_count=incorrect,
            total_count=total,
            items=item_results,
            by_category=by_category,
            by_difficulty=by_difficulty,
            usage=usage,
        )


async def evaluate_dataset(
    dataset: BaseDataset,
    model: ModelCallable,
    judge_model_config: "ModelConfig",
    grader_prompt: Optional[str] = None,
    parallel: bool = True,
    max_concurrency: int = 10,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> DatasetEvalResult:
    """Convenience function to evaluate a model against a dataset.

    This is the primary entry point for dataset evaluation.

    Args:
        dataset: The dataset to evaluate against
        model: Async callable that takes a query and returns an answer
        judge_model_config: ModelConfig for the judge LLM
        grader_prompt: Optional custom prompt for the correctness grader
        parallel: Whether to run operations in parallel
        max_concurrency: Maximum concurrent operations
        progress_callback: Optional callback(current, total, status)

    Returns:
        DatasetEvalResult with accuracy metrics

    Example:
        from tavily_agent_toolkit import ModelConfig, ModelObject
        from tavily_agent_toolkit.evals import evaluate_dataset, CSVDataset

        dataset = CSVDataset("queries.csv")

        async def my_model(query: str) -> str:
            return "my answer"

        result = await evaluate_dataset(
            dataset=dataset,
            model=my_model,
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )
        print(f"Accuracy: {result.accuracy:.1%}")
    """
    evaluator = DatasetEvaluator(
        judge_model_config=judge_model_config,
        grader_prompt=grader_prompt,
        parallel=parallel,
        max_concurrency=max_concurrency,
    )
    return await evaluator.evaluate(dataset, model, progress_callback)
