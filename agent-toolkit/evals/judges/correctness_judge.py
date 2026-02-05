"""Correctness judge for evaluating answer correctness."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .base import BaseJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


class CorrectnessGrade(BaseModel):
    """Schema for correctness grading output."""

    grade: Literal["CORRECT", "INCORRECT"] = Field(
        description="Whether the predicted answer is correct (CORRECT) or incorrect (INCORRECT)"
    )
    reasoning: str = Field(
        description="Brief explanation of why the answer is correct or incorrect"
    )


CORRECTNESS_GRADER_PROMPT = """You are an expert evaluator assessing whether a predicted answer correctly answers a question based on the expected answer.

Your task is to determine if the predicted answer is CORRECT or INCORRECT.

Grading criteria:
- CORRECT: The predicted answer conveys the same essential information as the expected answer. Minor differences in phrasing, additional context, or formatting are acceptable as long as the core answer is correct.
- INCORRECT: The predicted answer is factually wrong, contradicts the expected answer, misses key information, or fails to answer the question.

Important guidelines:
1. Focus on semantic correctness, not exact string matching
2. A more detailed answer that includes the expected information is still CORRECT
3. Partial answers that miss critical information should be INCORRECT
4. If the predicted answer says "I don't know" or similar, grade as INCORRECT
5. Be strict but fair - the answer must actually address what was asked

Return your grade (CORRECT or INCORRECT) along with brief reasoning."""


class CorrectnessJudge(BaseJudge):
    """Judge that grades answers as CORRECT or INCORRECT.

    This judge compares a predicted answer against an expected answer
    and determines if they are semantically equivalent.

    Example:
        judge = CorrectnessJudge(model_config=ModelConfig(...))
        result = await judge.judge(
            query="What is the capital of France?",
            expected_answer="Paris",
            predicted_answer="The capital of France is Paris.",
        )
        print(f"Grade: {result['grade']}")  # CORRECT
        print(f"Score: {result['score']}")  # 1.0
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        grader_prompt: Optional[str] = None,
    ):
        """Initialize the correctness judge.

        Args:
            model_config: ModelConfig for the judge LLM
            grader_prompt: Custom system prompt for grading (optional)
        """
        super().__init__(model_config)
        self.grader_prompt = grader_prompt or CORRECTNESS_GRADER_PROMPT

    async def judge(
        self,
        query: str,
        expected_answer: str,
        predicted_answer: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Judge whether the predicted answer is correct.

        Args:
            query: The original question/query
            expected_answer: The ground truth answer
            predicted_answer: The model's predicted answer
            **kwargs: Additional arguments (unused)

        Returns:
            Dict with:
                - grade: "CORRECT" or "INCORRECT"
                - score: 1.0 for CORRECT, 0.0 for INCORRECT
                - reasoning: Explanation of the grade
        """
        messages = [
            {"role": "system", "content": self.grader_prompt},
            {
                "role": "user",
                "content": f"""Please evaluate the following:

QUESTION: {query}

EXPECTED ANSWER: {expected_answer}

PREDICTED ANSWER: {predicted_answer}

Grade the predicted answer as CORRECT or INCORRECT based on whether it correctly answers the question.""",
            },
        ]

        result = await self.invoke_llm(messages, output_schema=CorrectnessGrade)

        # Convert grade to score
        score = 1.0 if result.grade == "CORRECT" else 0.0

        return {
            "grade": result.grade,
            "score": score,
            "reasoning": result.reasoning,
        }

    async def judge_batch(
        self,
        items: list[tuple[str, str, str]],
    ) -> list[dict[str, Any]]:
        """Judge multiple items sequentially.

        Args:
            items: List of (query, expected_answer, predicted_answer) tuples

        Returns:
            List of judgment results
        """
        results = []
        for query, expected, predicted in items:
            result = await self.judge(
                query=query,
                expected_answer=expected,
                predicted_answer=predicted,
            )
            results.append(result)
        return results
