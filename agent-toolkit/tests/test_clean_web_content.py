"""Test clean_raw_content with real web data from Tavily extract."""

import re
from pathlib import Path

from tavily import TavilyClient

# Read and execute clean_raw_content from local filesystem
_utils_path = Path(__file__).parent.parent / "utilities" / "utils.py"
_source = _utils_path.read_text()

# Extract just the clean_raw_content function
_start = _source.find("def clean_raw_content(")
_end = _source.find("\ndef ", _start + 1)
_func_source = _source[_start:_end]

# Execute to define the function
exec(compile(_func_source, "<clean_raw_content>", "exec"), globals())


def test_clean_web_content():
    """Extract content from URLs and compare before/after cleaning."""
    client = TavilyClient()
    
    urls = [
        "https://tavily.com/",
        "https://www.nvidia.com/en-us/",
        "https://finance.yahoo.com/quote/NVDA/",
        "https://www.reuters.com/technology/",
    ]
    
    print("\n" + "=" * 100)
    print("CLEAN_RAW_CONTENT COMPARISON TEST")
    print("=" * 100)
    
    response = client.extract(urls=urls)
    
    total_original = 0
    total_cleaned = 0
    
    for item in response["results"]:
        url = item.get("url", "Unknown URL")
        raw = item.get("raw_content", "")
        
        if not raw:
            print(f"\n[!] No content extracted from: {url}")
            continue
        
        cleaned = clean_raw_content(raw)
        
        orig_chars = len(raw)
        clean_chars = len(cleaned)
        orig_lines = len(raw.split('\n'))
        clean_lines = len(cleaned.split('\n'))
        reduction = 100 - (clean_chars / orig_chars * 100) if orig_chars > 0 else 0
        
        total_original += orig_chars
        total_cleaned += clean_chars
        
        print(f"\n{'-' * 100}")
        print(f"URL: {url}")
        print(f"{'-' * 100}")
        print(f"STATS:")
        print(f"   Original:  {orig_chars:>8,} chars | {orig_lines:>5,} lines")
        print(f"   Cleaned:   {clean_chars:>8,} chars | {clean_lines:>5,} lines")
        print(f"   Reduction: {reduction:>7.1f}%")
        
        # Show first 500 chars of original
        print(f"\n--- ORIGINAL (first 500 chars) ---")
        print(raw[:500])
        if len(raw) > 500:
            print("...")
        
        # Show first 1500 chars of cleaned
        print(f"\n--- CLEANED (first 1500 chars) ---")
        print(cleaned[:1500])
        if len(cleaned) > 1500:
            print("...")
    
    # Summary
    total_reduction = 100 - (total_cleaned / total_original * 100) if total_original > 0 else 0
    print(f"\n{'=' * 100}")
    print(f"TOTAL SUMMARY")
    print(f"{'=' * 100}")
    print(f"   Total Original:  {total_original:>10,} chars")
    print(f"   Total Cleaned:   {total_cleaned:>10,} chars")
    print(f"   Total Reduction: {total_reduction:>9.1f}%")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    test_clean_web_content()
