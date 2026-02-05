"""Base classes for evaluation datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class DatasetItem:
    """A single item in an evaluation dataset.

    Attributes:
        query: The input query/question to be answered
        expected_answer: The ground truth answer for evaluation
        category: Optional category for grouping results (e.g., "factual", "reasoning")
        difficulty: Optional difficulty level (e.g., "easy", "medium", "hard")
        metadata: Optional additional metadata as key-value pairs
    """

    query: str
    expected_answer: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "query": self.query,
            "expected_answer": self.expected_answer,
        }
        if self.category is not None:
            result["category"] = self.category
        if self.difficulty is not None:
            result["difficulty"] = self.difficulty
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BaseDataset(ABC):
    """Abstract base class for evaluation datasets.

    Datasets provide iteration over DatasetItem objects and support
    filtering by category or other attributes.

    Example:
        dataset = CSVDataset("my_queries.csv")
        for item in dataset:
            print(f"Query: {item.query}")

        # Filter by category
        factual_only = dataset.filter(category="factual")
    """

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterate over all items in the dataset."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        pass

    def filter(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> "FilteredDataset":
        """Filter dataset by category and/or difficulty.

        Args:
            category: Only include items with this category
            difficulty: Only include items with this difficulty level

        Returns:
            A new FilteredDataset containing only matching items
        """
        return FilteredDataset(self, category=category, difficulty=difficulty)

    def get_categories(self) -> list[str]:
        """Get all unique categories in the dataset.

        Returns:
            List of unique category strings (excludes None values)
        """
        categories = set()
        for item in self:
            if item.category is not None:
                categories.add(item.category)
        return sorted(categories)

    def get_difficulties(self) -> list[str]:
        """Get all unique difficulty levels in the dataset.

        Returns:
            List of unique difficulty strings (excludes None values)
        """
        difficulties = set()
        for item in self:
            if item.difficulty is not None:
                difficulties.add(item.difficulty)
        return sorted(difficulties)


class FilteredDataset(BaseDataset):
    """A dataset that filters another dataset by category/difficulty.

    This is returned by BaseDataset.filter() and provides a view
    over a subset of the original dataset.
    """

    def __init__(
        self,
        source: BaseDataset,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ):
        """Initialize filtered dataset.

        Args:
            source: The source dataset to filter
            category: Only include items with this category
            difficulty: Only include items with this difficulty
        """
        self._source = source
        self._category = category
        self._difficulty = difficulty
        # Pre-compute filtered items for efficient iteration
        self._items = self._compute_filtered_items()

    def _compute_filtered_items(self) -> list[DatasetItem]:
        """Filter source items based on criteria."""
        items = []
        for item in self._source:
            if self._category is not None and item.category != self._category:
                continue
            if self._difficulty is not None and item.difficulty != self._difficulty:
                continue
            items.append(item)
        return items

    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterate over filtered items."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return count of filtered items."""
        return len(self._items)


class InMemoryDataset(BaseDataset):
    """A dataset backed by an in-memory list of items.

    Useful for programmatically creating datasets or converting
    from other formats.

    Example:
        items = [
            DatasetItem(query="What is 2+2?", expected_answer="4", category="math"),
            DatasetItem(query="Capital of France?", expected_answer="Paris", category="geography"),
        ]
        dataset = InMemoryDataset(items)
    """

    def __init__(self, items: list[DatasetItem]):
        """Initialize with a list of items.

        Args:
            items: List of DatasetItem objects
        """
        self._items = items

    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterate over all items."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return total item count."""
        return len(self._items)

    @classmethod
    def from_dicts(cls, dicts: list[dict[str, Any]]) -> "InMemoryDataset":
        """Create dataset from list of dictionaries.

        Args:
            dicts: List of dicts with 'query', 'expected_answer', and optional fields

        Returns:
            InMemoryDataset instance
        """
        items = []
        for d in dicts:
            item = DatasetItem(
                query=d["query"],
                expected_answer=d["expected_answer"],
                category=d.get("category"),
                difficulty=d.get("difficulty"),
                metadata=d.get("metadata", {}),
            )
            items.append(item)
        return cls(items)
