"""Tests for the evaluation framework."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class TestEvalModels:
    """Test evaluation data models."""

    def test_eval_usage(self):
        """Test EvalUsage dataclass."""
        from tavily_agent_toolkit.evals import EvalUsage

        usage = EvalUsage()
        assert usage.judge_llm_calls == 0
        assert usage.judge_total_tokens == 0

        usage.add_call(input_tokens=100, output_tokens=50, response_time=1.5)
        assert usage.judge_llm_calls == 1
        assert usage.judge_input_tokens == 100
        assert usage.judge_output_tokens == 50
        assert usage.judge_total_tokens == 150

        # Test merge
        other = EvalUsage()
        other.add_call(input_tokens=200, output_tokens=100, response_time=2.0)
        usage.merge(other)
        assert usage.judge_llm_calls == 2
        assert usage.judge_total_tokens == 450

        # Test to_dict
        d = usage.to_dict()
        assert "judge_llm_calls" in d
        assert d["judge_total_tokens"] == 450

    def test_grounding_result(self):
        """Test GroundingResult dataclass."""
        from tavily_agent_toolkit.evals import GroundingResult, Claim, EvalUsage

        result = GroundingResult(
            grounding_ratio=0.8,
            citation_accuracy=0.9,
            unsupported_claims_count=2,
            supported_claims_count=8,
            total_claims=10,
            claim_details=[
                Claim(text="Test claim", is_supported=True, confidence=0.95)
            ],
            usage=EvalUsage(),
        )

        assert result.grounding_ratio == 0.8
        assert result.total_claims == 10

        d = result.to_dict()
        assert "grounding_ratio" in d
        assert len(d["claim_details"]) == 1

    def test_eval_result_overall_score(self):
        """Test EvalResult overall score computation."""
        from tavily_agent_toolkit.evals import (
            EvalResult,
            GroundingResult,
            RelevanceResult,
            EvalUsage,
        )

        result = EvalResult(query="Test query")
        result.grounding = GroundingResult(
            grounding_ratio=0.8,
            citation_accuracy=0.9,
            unsupported_claims_count=2,
            supported_claims_count=8,
            total_claims=10,
            usage=EvalUsage(),
        )
        result.relevance = RelevanceResult(
            source_relevance_score=0.7,
            answer_coverage=0.6,
            top_k_precision=0.8,
            usage=EvalUsage(),
        )

        score = result.compute_overall_score()
        assert 0 <= score <= 1
        assert result.overall_score == score

    def test_batch_eval_result_aggregates(self):
        """Test BatchEvalResult aggregate computation."""
        from tavily_agent_toolkit.evals import (
            BatchEvalResult,
            EvalResult,
            GroundingResult,
            EvalUsage,
        )

        results = []
        for ratio in [0.7, 0.8, 0.9]:
            r = EvalResult(query=f"Test {ratio}")
            r.grounding = GroundingResult(
                grounding_ratio=ratio,
                citation_accuracy=0.9,
                unsupported_claims_count=1,
                supported_claims_count=9,
                total_claims=10,
                usage=EvalUsage(),
            )
            results.append(r)

        batch = BatchEvalResult(results=results)
        batch.compute_aggregates()

        assert "mean_grounding" in batch.aggregate_scores
        assert batch.aggregate_scores["mean_grounding"] == pytest.approx(0.8, rel=0.01)


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
class TestMetricsIntegration:
    """Integration tests for metric computation (requires API keys)."""

    @pytest.mark.asyncio
    async def test_grounding_metrics_computation(self):
        """Test end-to-end grounding metrics computation."""
        from tavily_agent_toolkit import ModelConfig, ModelObject
        from tavily_agent_toolkit.evals import compute_grounding_metrics

        config = ModelConfig(
            model=ModelObject(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
            ),
            temperature=0.0,
        )

        report = "NVIDIA reported $35 billion in revenue for Q3 2024."
        sources = [
            {
                "url": "https://example.com/nvidia",
                "title": "NVIDIA Earnings",
                "content": "NVIDIA announced revenue of $35 billion for Q3 2024.",
            }
        ]

        result = await compute_grounding_metrics(
            report=report,
            sources=sources,
            judge_model_config=config,
        )

        assert 0 <= result.grounding_ratio <= 1
        assert result.total_claims >= 0
        assert result.usage.judge_llm_calls > 0

    @pytest.mark.asyncio
    async def test_relevance_metrics_computation(self):
        """Test end-to-end relevance metrics computation."""
        from tavily_agent_toolkit import ModelConfig, ModelObject
        from tavily_agent_toolkit.evals import compute_relevance_metrics

        config = ModelConfig(
            model=ModelObject(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
            ),
            temperature=0.0,
        )

        query = "What is NVIDIA's revenue?"
        sources = [
            {
                "url": "https://example.com/nvidia",
                "title": "NVIDIA Earnings",
                "content": "NVIDIA announced revenue of $35 billion for Q3 2024.",
            },
            {
                "url": "https://example.com/other",
                "title": "Other News",
                "content": "Unrelated content about weather.",
            },
        ]

        result = await compute_relevance_metrics(
            query=query,
            sources=sources,
            judge_model_config=config,
        )

        assert 0 <= result.source_relevance_score <= 1
        assert 0 <= result.answer_coverage <= 1
        assert len(result.per_source_scores) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
