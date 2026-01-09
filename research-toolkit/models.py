from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Type, TypedDict, Union

from pydantic import BaseModel


class OutputSchema(BaseModel):
    """Base class for output schemas passed to Tavily's research API.
    
    All fields MUST have a description in their Field() definition.
    
    Example:
        class MyOutputSchema(OutputSchema):
            topic: str = Field(..., description="The topic of the research")
            summary: str = Field(..., description="A brief summary of findings")
    """
    
    @classmethod
    def to_tavily_schema(cls) -> dict[str, Any]:
        """Convert to Tavily-compatible JSON Schema format.
        
        Returns a dict with 'properties' containing field definitions,
        each with 'type' and 'description'.
        
        Raises:
            ValueError: If any field is missing a description.
        """
        schema = cls.model_json_schema()
        properties = schema.get("properties", {})
        
        # Validate that all properties have descriptions
        missing_descriptions = [
            field_name for field_name, field_info in properties.items()
            if "description" not in field_info
        ]
        
        if missing_descriptions:
            raise ValueError(
                f"OutputSchema fields must have descriptions. "
                f"Missing descriptions for: {', '.join(missing_descriptions)}. "
                f"Use Field(..., description='...') for each field."
            )
        
        return {"properties": properties}


class _SearchResultOptional(TypedDict, total=False):
    """Optional fields for SearchResult."""
    score: float
    raw_content: Optional[str]
    published_date: Optional[str]
    favicon: Optional[str]


class SearchResult(_SearchResultOptional):
    """Individual search result from Tavily."""
    url: str
    title: str
    content: str


class ImageResult(TypedDict, total=False):
    """Image result from Tavily search."""
    url: str
    description: Optional[str]


class _SearchDedupResponseOptional(TypedDict, total=False):
    """Optional fields for SearchDedupResponse."""
    tavily_usage: dict[str, Any]
    response_time: float


class SearchDedupResponse(_SearchDedupResponseOptional):
    """Response from search_dedup function."""
    results: list[SearchResult]
    images: Optional[list[Union[str, ImageResult]]]
    answer: Optional[str]
    queries: list[str]


class WebSource(TypedDict):
    """Source reference with title and URL."""
    url: str
    title: Optional[str]


class HybridResearchResponse(TypedDict):
    """Response from hybrid_research function."""
    report: str | BaseModel
    web_sources: list[WebSource]


# =============================================================================
# Observability Types
# =============================================================================

@dataclass
class TavilyAPIResponse:
    """Response from a Tavily API call with timing and usage metadata."""
    data: dict[str, Any]
    response_time: float
    credits: int


@dataclass
class TavilyUsage:
    """Tracks Tavily API usage metrics."""
    total_credits: int = 0
    search_count: int = 0
    extract_count: int = 0
    crawl_count: int = 0
    search_response_time: float = 0.0
    extract_response_time: float = 0.0
    crawl_response_time: float = 0.0
    
    def add_search(self, credits: int, response_time: float) -> None:
        """Record a search API call."""
        self.total_credits += credits
        self.search_count += 1
        self.search_response_time += response_time
    
    def add_extract(self, credits: int, response_time: float) -> None:
        """Record an extract API call."""
        self.total_credits += credits
        self.extract_count += 1
        self.extract_response_time += response_time
    
    def add_crawl(self, credits: int, response_time: float) -> None:
        """Record a crawl API call."""
        self.total_credits += credits
        self.crawl_count += 1
        self.crawl_response_time += response_time
    
    def merge(self, other: "TavilyUsage") -> None:
        """Merge another TavilyUsage into this one."""
        self.total_credits += other.total_credits
        self.search_count += other.search_count
        self.extract_count += other.extract_count
        self.crawl_count += other.crawl_count
        self.search_response_time += other.search_response_time
        self.extract_response_time += other.extract_response_time
        self.crawl_response_time += other.crawl_response_time
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization. Only includes used API types."""
        result: dict[str, Any] = {"total_credits": self.total_credits}
        
        if self.search_count > 0:
            result["search_count"] = self.search_count
            result["search_response_time"] = round(self.search_response_time, 3)
        
        if self.extract_count > 0:
            result["extract_count"] = self.extract_count
            result["extract_response_time"] = round(self.extract_response_time, 3)
        
        if self.crawl_count > 0:
            result["crawl_count"] = self.crawl_count
            result["crawl_response_time"] = round(self.crawl_response_time, 3)
        
        return result


@dataclass
class LLMUsage:
    """Tracks LLM token usage metrics."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_call_count: int = 0
    llm_response_time: float = 0.0
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.total_input_tokens + self.total_output_tokens
    
    def add_call(self, input_tokens: int, output_tokens: int, response_time: float) -> None:
        """Record an LLM call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.llm_call_count += 1
        self.llm_response_time += response_time
    
    def merge(self, other: "LLMUsage") -> None:
        """Merge another LLMUsage into this one."""
        self.total_input_tokens += other.total_input_tokens
        self.total_output_tokens += other.total_output_tokens
        self.llm_call_count += other.llm_call_count
        self.llm_response_time += other.llm_response_time
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "llm_call_count": self.llm_call_count,
            "llm_response_time": round(self.llm_response_time, 3),
        }


@dataclass
class LLMResponse:
    """Response from an LLM call with usage metadata."""
    result: Any
    usage: "LLMUsage"


@dataclass
class ToolUsageStats:
    """Aggregated usage statistics for a tool call."""
    response_time: float = 0.0
    tavily: TavilyUsage = field(default_factory=TavilyUsage)
    llm: LLMUsage = field(default_factory=LLMUsage)
    internal_function_response_time: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization. Only includes used components."""
        result: dict[str, Any] = {"response_time": round(self.response_time, 3)}
        
        # Only include tavily if any API was called
        if self.tavily.total_credits > 0 or self.tavily.search_count > 0 or self.tavily.extract_count > 0 or self.tavily.crawl_count > 0:
            result["tavily"] = self.tavily.to_dict()
        
        # Only include llm if any LLM call was made
        if self.llm.llm_call_count > 0:
            result["llm"] = self.llm.to_dict()
        
        # Only include internal function time if used
        if self.internal_function_response_time > 0:
            result["internal_function_response_time"] = round(self.internal_function_response_time, 3)
        
        return result


# =============================================================================
# Model Configuration Types
# =============================================================================

# Supported model providers for init_chat_model
# Each requires its corresponding langchain integration package:
#   openai              -> langchain-openai
#   anthropic           -> langchain-anthropic
#   azure_openai        -> langchain-openai
#   azure_ai            -> langchain-azure-ai
#   google_vertexai     -> langchain-google-vertexai
#   google_genai        -> langchain-google-genai
#   bedrock             -> langchain-aws
#   bedrock_converse    -> langchain-aws
#   cohere              -> langchain-cohere
#   fireworks           -> langchain-fireworks
#   together            -> langchain-together
#   mistralai           -> langchain-mistralai
#   huggingface         -> langchain-huggingface
#   groq                -> langchain-groq
#   ollama              -> langchain-ollama
#   google_anthropic_vertex -> langchain-google-vertexai
#   deepseek            -> langchain-deepseek
#   ibm                 -> langchain-ibm
#   nvidia              -> langchain-nvidia-ai-endpoints
#   xai                 -> langchain-xai
#   perplexity          -> langchain-perplexity

ModelProvider = Literal[
    "openai",
    "anthropic",
    "azure_openai",
    "azure_ai",
    "google_vertexai",
    "google_genai",
    "bedrock",
    "bedrock_converse",
    "cohere",
    "fireworks",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "ollama",
    "google_anthropic_vertex",
    "deepseek",
    "ibm",
    "nvidia",
    "xai",
    "perplexity",
]

@dataclass
class ModelObject:
    """Configuration for initializing a chat model via langchain's init_chat_model.
    
    The model can be specified in two ways:
    1. Just `model` - provider will be inferred from model name prefix:
       - gpt-*, o1*, o3*  -> openai
       - claude*          -> anthropic
       - gemini*          -> google_vertexai
       - command*         -> cohere
       - mistral*         -> mistralai
       - deepseek*        -> deepseek
       - grok*            -> xai
       - sonar*           -> perplexity
       - amazon.*         -> bedrock
       - accounts/fireworks* -> fireworks
    
    2. Using "{provider}:{model}" format, e.g. "openai:gpt-5"
    
    3. Explicit `model` + `model_provider`
    
    Fallback Models:
        You can specify fallback_models in ModelConfig as a list of ModelObjects to try
        if the primary model fails.
        
        Example:
            ModelConfig(
                model=ModelObject(model="gpt-5", api_key="..."),
                fallback_models=[
                    ModelObject(model="claude-sonnet-4-5-20250929"),
                    ModelObject(model="gemini-2.5-flash")
                ],
                temperature=0.7
            )
        
        Retry behavior:
        - If fallback_models is provided: each model gets 1 attempt before moving to next
        - If no fallback_models: primary model gets 1 retry (2 attempts total)
    """
    
    # Required: Model name/ID (e.g., "gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-flash")
    # Can also use "{provider}:{model}" format (e.g., "openai:gpt-5")
    model: str
    model_provider: Optional[ModelProvider] = None    
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None


@dataclass
class ModelConfig:
    model: ModelObject
    fallback_models: Optional[list[ModelObject]] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = None

    def get_all_models(self) -> list[ModelObject]:
        """Return list of all models: primary followed by fallbacks."""
        models = [self.model]
        if self.fallback_models:
            models.extend(self.fallback_models)
        return models
    
    def to_init_kwargs(self, model_object: Optional["ModelObject"] = None) -> dict:
        """Convert to kwargs dict for init_chat_model, excluding None values.
        
        Args:
            model_object: If provided, use this ModelObject instead of self.model.
                         Used internally for fallback model initialization.
        """
        model_obj = model_object if model_object else self.model
        kwargs = {}
        kwargs["model"] = model_obj.model
        if model_obj.model_provider:
            kwargs["model_provider"] = model_obj.model_provider
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if model_obj.max_tokens is not None:
            kwargs["max_tokens"] = model_obj.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if model_obj.api_key:
            kwargs["api_key"] = model_obj.api_key
        return kwargs
