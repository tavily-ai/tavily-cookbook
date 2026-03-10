"""Base class for LLM-as-judge implementations."""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from pydantic import BaseModel

from ..models import EvalUsage

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig, ainvoke_with_fallback
except ImportError:
    try:
        from ...models import ModelConfig
        from ...utilities.utils import ainvoke_with_fallback
    except ImportError:
        ModelConfig = Any  # type: ignore
        ainvoke_with_fallback = None  # type: ignore


class BaseJudge(ABC):
    """Abstract base class for LLM-as-judge implementations.

    Subclasses should implement the `judge` method to perform specific
    evaluation tasks using an LLM.

    Example:
        class MyJudge(BaseJudge):
            async def judge(self, content: str, **kwargs) -> dict:
                # Use self.invoke_llm() for LLM calls
                result = await self.invoke_llm(
                    messages=[{"role": "user", "content": prompt}],
                    output_schema=MyOutputSchema,
                )
                return result

    Attributes:
        model_config: ModelConfig for the judge LLM
        usage: EvalUsage tracking LLM calls
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        system_prompt: Optional[str] = None,
    ):
        """Initialize the judge.

        Args:
            model_config: ModelConfig for the judge LLM
            system_prompt: Optional system prompt for all judge calls
        """
        self.model_config = model_config
        self.system_prompt = system_prompt
        self.usage = EvalUsage()

    def reset_usage(self) -> None:
        """Reset usage tracking."""
        self.usage = EvalUsage()

    async def invoke_llm(
        self,
        messages: list[dict],
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the LLM with usage tracking.

        Args:
            messages: List of message dicts for the LLM
            output_schema: Optional Pydantic model for structured output
            **kwargs: Additional kwargs passed to ainvoke_with_fallback

        Returns:
            LLM response (structured if output_schema provided)
        """
        if ainvoke_with_fallback is None:
            raise ImportError("ainvoke_with_fallback not available")

        # Prepend system prompt if configured
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        start_time = time.perf_counter()
        response = await ainvoke_with_fallback(
            self.model_config,
            messages,
            output_schema=output_schema,
            return_usage=True,
            **kwargs,
        )
        elapsed = time.perf_counter() - start_time

        # Track usage
        self.usage.add_call(
            input_tokens=response.usage.total_input_tokens,
            output_tokens=response.usage.total_output_tokens,
            response_time=elapsed,
        )

        return response.result

    @abstractmethod
    async def judge(self, **kwargs: Any) -> Any:
        """Perform the judgment task.

        Subclasses must implement this method.

        Returns:
            Judgment result (structure depends on subclass)
        """
        pass

    def get_usage(self) -> EvalUsage:
        """Get current usage statistics.

        Returns:
            Copy of current EvalUsage
        """
        return EvalUsage(
            judge_llm_calls=self.usage.judge_llm_calls,
            judge_input_tokens=self.usage.judge_input_tokens,
            judge_output_tokens=self.usage.judge_output_tokens,
            eval_response_time=self.usage.eval_response_time,
        )
