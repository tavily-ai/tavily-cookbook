"""CSV-based evaluation dataset."""

import json
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from .base import BaseDataset, DatasetItem, InMemoryDataset


class CSVDataset(BaseDataset):
    """Load evaluation datasets from CSV files.

    Expected CSV format:
    - Required columns: `query`, `expected_answer`
    - Optional columns: `category`, `difficulty`, `metadata` (JSON string)

    Example CSV:
        query,expected_answer,category,difficulty
        "What is 2+2?","4","math","easy"
        "Capital of France?","Paris","geography","easy"
        "Explain quantum entanglement","Quantum entanglement is...","physics","hard"

    Example with metadata:
        query,expected_answer,category,metadata
        "Who won Super Bowl 2024?","Kansas City Chiefs","sports","{\"date\": \"2024-02-11\"}"

    Example:
        dataset = CSVDataset("my_queries.csv")
        print(f"Loaded {len(dataset)} items")

        for item in dataset:
            print(f"Q: {item.query}")
            print(f"A: {item.expected_answer}")
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        query_column: str = "query",
        answer_column: str = "expected_answer",
        category_column: str = "category",
        difficulty_column: str = "difficulty",
        metadata_column: str = "metadata",
        encoding: str = "utf-8",
    ):
        """Load dataset from a CSV file.

        Args:
            file_path: Path to the CSV file
            query_column: Name of the column containing queries
            answer_column: Name of the column containing expected answers
            category_column: Name of the optional category column
            difficulty_column: Name of the optional difficulty column
            metadata_column: Name of the optional metadata column (JSON)
            encoding: File encoding (default: utf-8)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing
        """
        self._file_path = Path(file_path)
        self._query_column = query_column
        self._answer_column = answer_column
        self._category_column = category_column
        self._difficulty_column = difficulty_column
        self._metadata_column = metadata_column
        self._encoding = encoding

        # Load and validate
        self._items = self._load_csv()

    def _load_csv(self) -> list[DatasetItem]:
        """Load items from CSV file."""
        import csv

        if not self._file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._file_path}")

        items = []
        with open(self._file_path, "r", encoding=self._encoding, newline="") as f:
            reader = csv.DictReader(f)

            # Validate required columns
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty")

            fieldnames = list(reader.fieldnames)
            if self._query_column not in fieldnames:
                raise ValueError(
                    f"Required column '{self._query_column}' not found. "
                    f"Available columns: {fieldnames}"
                )
            if self._answer_column not in fieldnames:
                raise ValueError(
                    f"Required column '{self._answer_column}' not found. "
                    f"Available columns: {fieldnames}"
                )

            # Check for optional columns
            has_category = self._category_column in fieldnames
            has_difficulty = self._difficulty_column in fieldnames
            has_metadata = self._metadata_column in fieldnames

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
                query = row.get(self._query_column, "").strip()
                expected_answer = row.get(self._answer_column, "").strip()

                if not query:
                    raise ValueError(f"Empty query at row {row_num}")
                if not expected_answer:
                    raise ValueError(f"Empty expected_answer at row {row_num}")

                # Parse optional fields
                category = None
                if has_category:
                    cat_val = row.get(self._category_column, "").strip()
                    category = cat_val if cat_val else None

                difficulty = None
                if has_difficulty:
                    diff_val = row.get(self._difficulty_column, "").strip()
                    difficulty = diff_val if diff_val else None

                metadata: dict[str, Any] = {}
                if has_metadata:
                    meta_val = row.get(self._metadata_column, "").strip()
                    if meta_val:
                        try:
                            metadata = json.loads(meta_val)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSON in metadata at row {row_num}: {e}"
                            )

                item = DatasetItem(
                    query=query,
                    expected_answer=expected_answer,
                    category=category,
                    difficulty=difficulty,
                    metadata=metadata,
                )
                items.append(item)

        return items

    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterate over all items."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return total item count."""
        return len(self._items)

    @property
    def file_path(self) -> Path:
        """Get the source file path."""
        return self._file_path

    @classmethod
    def from_dataframe(
        cls,
        df: "Any",  # pandas.DataFrame
        query_column: str = "query",
        answer_column: str = "expected_answer",
        category_column: str = "category",
        difficulty_column: str = "difficulty",
        metadata_column: str = "metadata",
    ) -> InMemoryDataset:
        """Create dataset from a pandas DataFrame.

        Args:
            df: pandas DataFrame with query and answer columns
            query_column: Name of the query column
            answer_column: Name of the expected answer column
            category_column: Name of the optional category column
            difficulty_column: Name of the optional difficulty column
            metadata_column: Name of the optional metadata column

        Returns:
            InMemoryDataset instance (not CSVDataset since there's no file)

        Example:
            import pandas as pd
            df = pd.read_csv("queries.csv")
            dataset = CSVDataset.from_dataframe(df)
        """
        # Validate required columns
        if query_column not in df.columns:
            raise ValueError(
                f"Required column '{query_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        if answer_column not in df.columns:
            raise ValueError(
                f"Required column '{answer_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        items = []
        for idx, row in df.iterrows():
            query = str(row[query_column]).strip()
            expected_answer = str(row[answer_column]).strip()

            if not query:
                raise ValueError(f"Empty query at index {idx}")
            if not expected_answer:
                raise ValueError(f"Empty expected_answer at index {idx}")

            # Parse optional fields
            category = None
            if category_column in df.columns:
                cat_val = row.get(category_column)
                if cat_val is not None and str(cat_val).strip():
                    category = str(cat_val).strip()

            difficulty = None
            if difficulty_column in df.columns:
                diff_val = row.get(difficulty_column)
                if diff_val is not None and str(diff_val).strip():
                    difficulty = str(diff_val).strip()

            metadata: dict[str, Any] = {}
            if metadata_column in df.columns:
                meta_val = row.get(metadata_column)
                if meta_val is not None and str(meta_val).strip():
                    try:
                        if isinstance(meta_val, dict):
                            metadata = meta_val
                        else:
                            metadata = json.loads(str(meta_val))
                    except json.JSONDecodeError:
                        pass  # Ignore invalid JSON

            item = DatasetItem(
                query=query,
                expected_answer=expected_answer,
                category=category,
                difficulty=difficulty,
                metadata=metadata,
            )
            items.append(item)

        return InMemoryDataset(items)

    def to_dataframe(self) -> "Any":  # pandas.DataFrame
        """Convert dataset to a pandas DataFrame.

        Returns:
            pandas DataFrame with all dataset columns

        Example:
            dataset = CSVDataset("queries.csv")
            df = dataset.to_dataframe()
            df.to_csv("output.csv", index=False)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        rows = []
        for item in self._items:
            row = {
                "query": item.query,
                "expected_answer": item.expected_answer,
                "category": item.category,
                "difficulty": item.difficulty,
            }
            if item.metadata:
                row["metadata"] = json.dumps(item.metadata)
            rows.append(row)

        return pd.DataFrame(rows)
