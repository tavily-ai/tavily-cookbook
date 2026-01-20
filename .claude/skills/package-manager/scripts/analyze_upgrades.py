#!/usr/bin/env python3
"""
Package Upgrade Analyzer

Analyzes outdated packages using Tavily Research API and provides structured
recommendations. Advisory only - never auto-upgrades packages.

Output includes:
- Risk level (LOW/MEDIUM/HIGH)
- Breaking changes and migration steps
- Code impact (files using affected APIs)
- Upgrade command
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Configure logging to stderr so it doesn't interleave with stdout results
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


# Structured output schema for Tavily Research API
UPGRADE_RESEARCH_SCHEMA = {
    "properties": {
        "summary": {
            "type": "string",
            "description": "Brief summary of the upgrade path and key considerations"
        },
        "breaking_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "change": {
                        "type": "string",
                        "description": "Description of the breaking change"
                    },
                    "affected_api": {
                        "type": "string",
                        "description": "The specific function, class, or module affected"
                    },
                    "migration": {
                        "type": "string",
                        "description": "How to migrate or fix this breaking change"
                    }
                },
                "required": ["change", "affected_api", "migration"]
            },
            "description": "List of breaking changes between versions"
        },
        "deprecated_apis": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of deprecated function/class names"
        },
        "risk_level": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH"],
            "description": "LOW = safe upgrade, MEDIUM = review recommended, HIGH = significant changes required"
        },
        "risk_explanation": {
            "type": "string",
            "description": "Explanation for the risk level assessment"
        },
        "upgrade_command": {
            "type": "string",
            "description": "The exact command to upgrade (e.g., pip install package==version)"
        }
    },
    "required": ["summary", "breaking_changes", "risk_level", "risk_explanation", "upgrade_command"]
}


class PackageAnalyzer:
    """Analyzes packages and identifies outdated dependencies."""

    EXCLUDED_DIRS = {"node_modules", "venv", ".venv", "__pycache__", ".git", "dist", "build", ".tox", "env"}

    def __init__(self, project_path: Optional[str] = None):
        if project_path:
            self.project_path = Path(project_path).resolve()
            if not self.project_path.exists():
                raise ValueError(f"Project path does not exist: {project_path}")
            if not self.project_path.is_dir():
                raise ValueError(f"Project path is not a directory: {project_path}")
        else:
            self.project_path = Path.cwd()
        self.package_manager = None

    def detect_package_manager(self) -> Optional[str]:
        """Detect which package manager the project uses."""
        # Check Python package managers
        python_indicators = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]
        for indicator in python_indicators:
            if (self.project_path / indicator).exists():
                self.package_manager = "pip"
                return "pip"

        # Check JS package managers (order matters: prefer yarn/pnpm if lockfiles exist)
        if (self.project_path / "pnpm-lock.yaml").exists():
            self.package_manager = "pnpm"
            return "pnpm"

        if (self.project_path / "yarn.lock").exists():
            self.package_manager = "yarn"
            return "yarn"

        if (self.project_path / "package.json").exists():
            self.package_manager = "npm"
            return "npm"

        return None

    def get_outdated_packages(self) -> list[dict]:
        """Get list of outdated packages."""
        if not self.package_manager:
            self.detect_package_manager()

        if self.package_manager == "pip":
            return self._get_pip_outdated()
        elif self.package_manager in ("npm", "yarn", "pnpm"):
            return self._get_js_outdated()
        return []

    def _find_venv_python(self) -> Optional[str]:
        """Find Python executable in a virtual environment within the project.
        
        First checks common venv directory names (venv, .venv, env), then searches
        all subdirectories for venv structure. Returns the path to the Python executable
        if found, None otherwise.
        """
        def _check_venv_path(venv_path: Path) -> Optional[str]:
            """Check if a path is a valid virtual environment and return Python executable."""
            if not venv_path.exists() or not venv_path.is_dir():
                return None
            
            # Check for pyvenv.cfg (standard marker for Python venv)
            pyvenv_cfg = venv_path / "pyvenv.cfg"
            has_pyvenv_cfg = pyvenv_cfg.exists()
            
            # Check for Python executable
            if sys.platform == "win32":
                venv_python = venv_path / "Scripts" / "python.exe"
                activate_script = venv_path / "Scripts" / "activate.bat"
            else:
                venv_python = venv_path / "bin" / "python"
                activate_script = venv_path / "bin" / "activate"
            
            # Valid venv must have Python executable and either pyvenv.cfg or activate script
            if venv_python.exists() and venv_python.is_file():
                if has_pyvenv_cfg or activate_script.exists():
                    return str(venv_python)
            
            return None
        
        # First, check common venv directory names (even if in EXCLUDED_DIRS)
        common_venv_names = ["venv", ".venv", "env", ".env"]
        for venv_name in common_venv_names:
            venv_path = self.project_path / venv_name
            python_exe = _check_venv_path(venv_path)
            if python_exe:
                logger.debug(f"Found virtual environment: {venv_name}")
                return python_exe
        
        # Then check all other subdirectories (excluding common names we already checked)
        try:
            for item in self.project_path.iterdir():
                if not item.is_dir():
                    continue
                
                # Skip if it's a common venv name we already checked
                if item.name in common_venv_names:
                    continue
                
                # Skip other excluded directories
                if item.name in self.EXCLUDED_DIRS:
                    continue
                
                python_exe = _check_venv_path(item)
                if python_exe:
                    logger.debug(f"Found virtual environment: {item.name}")
                    return python_exe
        except (PermissionError, OSError) as e:
            logger.debug(f"Could not iterate project directory: {e}")
        
        return None

    def _get_pip_outdated(self) -> list[dict]:
        """Get outdated pip packages from the project's environment."""
        # Try to find and use the project's virtual environment
        python_exe = sys.executable
        venv_python = self._find_venv_python()
        if venv_python:
            python_exe = venv_python
            logger.debug(f"Using virtual environment Python: {python_exe}")
        else:
            logger.debug(f"Using system Python: {python_exe}")

        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_path,
                timeout=30,  # Prevent hanging
            )
            if result.returncode != 0:
                logger.warning(f"pip list failed (exit code {result.returncode}): {result.stderr}")
                return []

            if not result.stdout or not result.stdout.strip():
                logger.debug("pip list returned empty output")
                return []

            packages = json.loads(result.stdout)
            if not isinstance(packages, list):
                logger.warning(f"pip list returned unexpected format: {type(packages)}")
                return []

            return [
                {
                    "name": pkg["name"],
                    "current_version": pkg["version"],
                    "latest_version": pkg["latest_version"],
                    "package_manager": "pip",
                }
                for pkg in packages
                if isinstance(pkg, dict) and "name" in pkg and "version" in pkg and "latest_version" in pkg
            ]
        except subprocess.TimeoutExpired:
            logger.error("pip list command timed out")
            return []
        except FileNotFoundError:
            logger.error(f"Python executable not found: {python_exe}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pip output as JSON: {e}")
            logger.debug(f"pip output: {result.stdout[:200] if result.stdout else 'empty'}")
            return []
        except Exception as e:
            logger.error(f"Error getting pip packages: {e}", exc_info=True)
            return []

    def _get_js_outdated(self) -> list[dict]:
        """Get outdated JS packages (npm/yarn/pnpm)."""
        # Determine command based on package manager
        if self.package_manager == "pnpm":
            cmd = ["pnpm", "outdated", "--format=json"]
        elif self.package_manager == "yarn":
            cmd = ["yarn", "outdated", "--json"]
        else:
            cmd = ["npm", "outdated", "--json"]

        # Check if command exists
        if not shutil.which(cmd[0]):
            logger.warning(f"{cmd[0]} not found in PATH")
            return []

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_path,
                timeout=60,  # Prevent hanging (longer than pip due to network calls)
            )
            # npm outdated returns exit code 1 when packages are outdated
            # This is expected behavior, not an error
            output = result.stdout if result.stdout else "{}"

            if self.package_manager == "yarn":
                # Yarn outputs newline-delimited JSON
                return self._parse_yarn_outdated(output)
            elif self.package_manager == "pnpm":
                return self._parse_pnpm_outdated(output)
            else:
                return self._parse_npm_outdated(output)

        except subprocess.TimeoutExpired:
            logger.error(f"{self.package_manager} outdated command timed out after 60 seconds")
            return []
        except FileNotFoundError:
            logger.error(f"{cmd[0]} executable not found")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {self.package_manager} output as JSON: {e}")
            logger.debug(f"{self.package_manager} output: {result.stdout[:200] if 'result' in locals() and result.stdout else 'empty'}")
            return []
        except Exception as e:
            logger.error(f"Error getting {self.package_manager} packages: {e}", exc_info=True)
            return []

    def _parse_npm_outdated(self, output: str) -> list[dict]:
        """Parse npm outdated --json output."""
        packages_dict = json.loads(output) if output.strip() else {}
        return [
            {
                "name": name,
                "current_version": info.get("current", "unknown"),
                "latest_version": info.get("latest", "unknown"),
                "package_manager": "npm",
            }
            for name, info in packages_dict.items()
        ]

    def _parse_yarn_outdated(self, output: str) -> list[dict]:
        """Parse yarn outdated --json output (newline-delimited JSON)."""
        packages = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "table":
                    # Yarn v1 format - check that data["data"] is a dict before accessing
                    data_dict = data.get("data")
                    if isinstance(data_dict, dict):
                        body = data_dict.get("body", [])
                        if isinstance(body, list):
                            for row in body:
                                if isinstance(row, list) and len(row) >= 4:
                                    packages.append({
                                        "name": row[0],
                                        "current_version": row[1],
                                        "latest_version": row[3],
                                        "package_manager": "yarn",
                                    })
            except (json.JSONDecodeError, AttributeError, TypeError, KeyError, IndexError):
                # Skip malformed lines - continue processing other lines
                continue
        return packages

    def _parse_pnpm_outdated(self, output: str) -> list[dict]:
        """Parse pnpm outdated --format=json output."""
        packages = []
        try:
            data = json.loads(output) if output.strip() else {}
            # pnpm outputs { "packageName": { "current": "x", "latest": "y" }, ... }
            for name, info in data.items():
                if isinstance(info, dict):
                    packages.append({
                        "name": name,
                        "current_version": info.get("current", "unknown"),
                        "latest_version": info.get("latest", "unknown"),
                        "package_manager": "pnpm",
                    })
        except json.JSONDecodeError:
            pass
        return packages

    def find_package_usage(self, package_name: str) -> list[dict]:
        """Find where a package is imported in the codebase."""
        usages = []

        if self.package_manager == "pip":
            extensions = [".py"]
            import_patterns = [
                rf"^\s*import\s+{re.escape(package_name)}\b",
                rf"^\s*from\s+{re.escape(package_name)}\b",
            ]
        elif self.package_manager in ("npm", "yarn", "pnpm"):
            extensions = [".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"]
            import_patterns = [
                rf"require\s*\(\s*['\"]({re.escape(package_name)})['\"]",
                rf"from\s+['\"]({re.escape(package_name)})['\"]",
                rf"import\s+['\"]({re.escape(package_name)})['\"]",
            ]
        else:
            return usages

        for ext in extensions:
            for file_path in self.project_path.rglob(f"*{ext}"):
                if any(part in self.EXCLUDED_DIRS for part in file_path.relative_to(self.project_path).parts):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    for line_num, line in enumerate(content.split("\n"), 1):
                        for pattern in import_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                usages.append({
                                    "file": str(file_path.relative_to(self.project_path)),
                                    "line": line_num,
                                    "content": line.strip(),
                                })
                                break  # Avoid duplicate matches for same line
                except Exception as e:
                    logger.debug(f"Could not read {file_path}: {e}")
                    continue

        return usages

    def find_api_usage(self, package_name: str, api_patterns: list[str]) -> list[dict]:
        """Find where specific APIs from a package are used in the codebase.

        This searches for actual API usage, not just imports.
        """
        usages = []

        if not api_patterns:
            return usages

        if self.package_manager == "pip":
            extensions = [".py"]
        elif self.package_manager in ("npm", "yarn", "pnpm"):
            extensions = [".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"]
        else:
            return usages

        # Build regex patterns for API usage
        compiled_patterns = []
        for api in api_patterns:
            if not api:
                continue
            # Match the API name as a word (not part of another identifier)
            # This catches: np.string_, obj.string_, string_(, etc.
            pattern = rf"\b{re.escape(api)}\b"
            try:
                compiled_patterns.append((api, re.compile(pattern)))
            except re.error as e:
                logger.debug(f"Invalid regex pattern for API '{api}': {e}")
                continue

        if not compiled_patterns:
            return usages

        for ext in extensions:
            for file_path in self.project_path.rglob(f"*{ext}"):
                if any(part in self.EXCLUDED_DIRS for part in file_path.relative_to(self.project_path).parts):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.split("\n")

                    for line_num, line in enumerate(lines, 1):
                        # Skip import lines - we want actual usage
                        stripped = line.strip()
                        if stripped.startswith(("import ", "from ", "require(", "export ")):
                            continue

                        for api_name, pattern in compiled_patterns:
                            if pattern.search(line):
                                usages.append({
                                    "file": str(file_path.relative_to(self.project_path)),
                                    "line": line_num,
                                    "content": stripped[:100],  # Truncate long lines
                                    "matched_api": api_name,
                                })
                                break  # One match per line is enough

                except Exception as e:
                    logger.debug(f"Could not read {file_path}: {e}")
                    continue

        return usages


def validate_research_response(data: dict) -> dict:
    """Validate and normalize Tavily research response against expected schema.

    Returns a normalized dict with all expected fields, using defaults for missing ones.
    """
    validated = {
        "summary": "",
        "breaking_changes": [],
        "deprecated_apis": [],
        "risk_level": "UNKNOWN",
        "risk_explanation": "",
        "upgrade_command": "",
    }

    if not isinstance(data, dict):
        logger.warning("Research response is not a dict, using defaults")
        return validated

    # Validate summary
    if isinstance(data.get("summary"), str):
        validated["summary"] = data["summary"]

    # Validate risk_level
    risk_value = data.get("risk_level")
    if isinstance(risk_value, str):
        risk = risk_value.upper()
        if risk in ("LOW", "MEDIUM", "HIGH"):
            validated["risk_level"] = risk
        else:
            logger.warning(f"Invalid risk_level '{risk_value}', using UNKNOWN")
    elif risk_value is not None:
        logger.warning(f"risk_level is not a string: {type(risk_value).__name__}, using UNKNOWN")

    # Validate risk_explanation
    if isinstance(data.get("risk_explanation"), str):
        validated["risk_explanation"] = data["risk_explanation"]

    # Validate upgrade_command
    if isinstance(data.get("upgrade_command"), str):
        validated["upgrade_command"] = data["upgrade_command"]

    # Validate breaking_changes array
    if isinstance(data.get("breaking_changes"), list):
        for bc in data["breaking_changes"]:
            if isinstance(bc, dict):
                # Handle None values: .get() returns None if key exists with None value
                change = bc.get("change") or ""
                affected_api = bc.get("affected_api") or ""
                migration = bc.get("migration") or ""
                # Ensure all values are strings
                validated["breaking_changes"].append({
                    "change": str(change) if change is not None else "",
                    "affected_api": str(affected_api) if affected_api is not None else "",
                    "migration": str(migration) if migration is not None else "",
                })

    # Validate deprecated_apis array
    if isinstance(data.get("deprecated_apis"), list):
        for api in data["deprecated_apis"]:
            if isinstance(api, str) and api:
                validated["deprecated_apis"].append(api)

    return validated


def research_package(
    client: "TavilyClient",
    package_name: str,
    current_version: str,
    target_version: str,
    package_manager: str = "pip",
    poll_interval: int = 5,
    max_wait: int = 180,
) -> dict:
    """Research a package upgrade using Tavily Research API.

    Args:
        client: TavilyClient instance
        package_name: Name of the package
        current_version: Currently installed version
        target_version: Target version to upgrade to
        package_manager: pip, npm, yarn, or pnpm
        poll_interval: Seconds between status checks (default: 5)
        max_wait: Maximum seconds to wait for research (default: 180)

    Returns:
        Dict with status, package info, and research data
    """
    prompt = f"""Research the upgrade path for the {package_manager} package "{package_name}" from version {current_version} to {target_version}.

Analyze:
1. All breaking changes, removed APIs, and changed function signatures between these versions
2. Deprecated APIs that still work but should be updated
3. Risk assessment for upgrade (LOW/MEDIUM/HIGH)

Be specific about exact function names, class names, and module paths that have changed.
Focus on practical migration steps developers need to take."""

    logger.info(f"Researching {package_name} {current_version} → {target_version}...")

    try:
        result = client.research(
            input=prompt,
            model="mini",
            output_schema=UPGRADE_RESEARCH_SCHEMA,
        )
        request_id = result.get("request_id")
        if not request_id:
            logger.error(f"No request_id returned for {package_name}")
            return {"status": "failed", "package": package_name, "error": "No request_id"}
    except Exception as e:
        logger.error(f"Failed to start research for {package_name}: {e}")
        return {"status": "failed", "package": package_name, "error": str(e)}

    start_time = time.monotonic()
    while (time.monotonic() - start_time) < max_wait:
        try:
            response = client.get_research(request_id)
            # Validate response is a dict before accessing it
            if not isinstance(response, dict):
                logger.error(f"Invalid response type for {package_name}: {type(response).__name__}")
                return {"status": "failed", "package": package_name, "error": f"Invalid response type: {type(response).__name__}"}
        except Exception as e:
            logger.error(f"Failed to get research status for {package_name}: {e}")
            return {"status": "failed", "package": package_name, "error": str(e)}

        status = response.get("status", "unknown")

        if status == "completed":
            content = response.get("content")
            if isinstance(content, dict):
                raw_data = content
            elif isinstance(content, str):
                try:
                    raw_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse research content as JSON for {package_name}")
                    raw_data = {"raw_content": content}
            else:
                raw_data = {}

            # Validate and normalize the response
            validated_data = validate_research_response(raw_data)

            return {
                "status": "completed",
                "package": package_name,
                "current_version": current_version,
                "target_version": target_version,
                "data": validated_data,
            }

        elif status == "failed":
            error_msg = response.get("error", "Unknown error")
            logger.error(f"Research failed for {package_name}: {error_msg}")
            return {"status": "failed", "package": package_name, "error": error_msg}

        time.sleep(poll_interval)

    elapsed = time.monotonic() - start_time
    logger.warning(f"Research timed out for {package_name} after {elapsed:.1f}s")
    return {"status": "timeout", "package": package_name}


def analyze_packages(
    project_path: Optional[str] = None,
    specific_packages: Optional[list] = None,
    tavily_client: Optional["TavilyClient"] = None,
    poll_interval: int = 5,
    max_wait: int = 180,
) -> list[dict]:
    """Analyze packages and return structured recommendations.

    Args:
        project_path: Path to the project directory (default: current directory)
        specific_packages: Optional list of package names to analyze (default: all outdated)
        tavily_client: TavilyClient instance for research (optional)
        poll_interval: Seconds between research status checks
        max_wait: Maximum seconds to wait for research per package

    Returns:
        List of package analysis results
    """
    try:
        analyzer = PackageAnalyzer(project_path)
    except ValueError as e:
        logger.error(str(e))
        return []
    manager = analyzer.detect_package_manager()

    if not manager:
        logger.error("Could not detect package manager (pip/npm/yarn/pnpm)")
        return []

    logger.info(f"Detected package manager: {manager}")
    logger.info("Finding outdated packages...")

    outdated = analyzer.get_outdated_packages()

    if specific_packages:
        specific_lower = [s.lower() for s in specific_packages]
        outdated = [p for p in outdated if p["name"].lower() in specific_lower]

    if not outdated:
        logger.info("No outdated packages found!")
        return []

    logger.info(f"Found {len(outdated)} outdated package(s)")

    results = []

    for pkg in outdated:
        result = {
            "package": pkg["name"],
            "current_version": pkg["current_version"],
            "latest_version": pkg["latest_version"],
            "package_manager": pkg["package_manager"],
            "risk_level": "UNKNOWN",
            "risk_explanation": "",
            "summary": "",
            "breaking_changes": [],
            "deprecated_apis": [],
            "code_impact": [],
            "import_locations": [],
            "upgrade_command": "",
        }

        # Research via Tavily if available
        if tavily_client:
            research = research_package(
                tavily_client,
                pkg["name"],
                pkg["current_version"],
                pkg["latest_version"],
                pkg["package_manager"],
                poll_interval=poll_interval,
                max_wait=max_wait,
            )

            if research.get("status") == "completed":
                data = research.get("data", {})
                result["risk_level"] = data.get("risk_level", "UNKNOWN")
                result["risk_explanation"] = data.get("risk_explanation", "")
                result["summary"] = data.get("summary", "")
                result["breaking_changes"] = data.get("breaking_changes", [])
                result["deprecated_apis"] = data.get("deprecated_apis", [])
                result["upgrade_command"] = data.get("upgrade_command", "")

        # Set default upgrade command if not set
        if not result["upgrade_command"]:
            if pkg["package_manager"] == "pip":
                result["upgrade_command"] = f"pip install {pkg['name']}=={pkg['latest_version']}"
            elif pkg["package_manager"] == "yarn":
                result["upgrade_command"] = f"yarn add {pkg['name']}@{pkg['latest_version']}"
            elif pkg["package_manager"] == "pnpm":
                result["upgrade_command"] = f"pnpm add {pkg['name']}@{pkg['latest_version']}"
            else:
                result["upgrade_command"] = f"npm install {pkg['name']}@{pkg['latest_version']}"

        # Find where the package is imported
        result["import_locations"] = analyzer.find_package_usage(pkg["name"])

        # Build list of API patterns to search for (breaking + deprecated)
        api_patterns = list(result["deprecated_apis"])
        for bc in result["breaking_changes"]:
            api = bc.get("affected_api", "")
            if api:
                # Extract the last part of the API path (e.g., "np.string_" -> "string_")
                api_name = api.split(".")[-1]
                if api_name and api_name not in api_patterns:
                    api_patterns.append(api_name)

        # Find actual API usage in code (not just imports)
        if api_patterns:
            result["code_impact"] = analyzer.find_api_usage(pkg["name"], api_patterns)

        results.append(result)

    return results


def print_results(results: list[dict]):
    """Print results in a readable format."""

    print("\n" + "=" * 70)
    print("PACKAGE UPGRADE ANALYSIS")
    print("=" * 70)

    for r in results:
        risk_indicator = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(r["risk_level"], "⚪")

        print(f"\n{risk_indicator} {r['package']} {r['current_version']} → {r['latest_version']}")
        print(f"   Risk: {r['risk_level']}")

        summary = r.get("summary")
        if summary and isinstance(summary, str):
            # Wrap summary to ~70 chars
            summary_str = summary[:150] + "..." if len(summary) > 150 else summary
            print(f"   Summary: {summary_str}")

        risk_explanation = r.get("risk_explanation")
        if risk_explanation and isinstance(risk_explanation, str):
            risk_explanation_str = risk_explanation[:100] + "..." if len(risk_explanation) > 100 else risk_explanation
            print(f"   {risk_explanation_str}")

        if r.get("breaking_changes"):
            print(f"\n   Breaking Changes:")
            for bc in r["breaking_changes"][:5]:
                affected_api = bc.get('affected_api') or 'Unknown'
                change = bc.get('change') or ''
                change_str = change[:60] if change else ''
                print(f"     • {affected_api}: {change_str}")
                migration = bc.get("migration")
                if migration:
                    migration_str = migration[:60] if migration else ''
                    print(f"       → {migration_str}")

        if r.get("code_impact"):
            print(f"\n   Code Impact (affected API usage found):")
            for impact in r["code_impact"][:10]:
                matched = impact.get("matched_api", "")
                matched_str = f" [{matched}]" if matched else ""
                print(f"     • {impact['file']}:{impact['line']}{matched_str}")
                if impact.get("content"):
                    print(f"       {impact['content'][:70]}")

        if r.get("import_locations") and not r.get("code_impact"):
            # Only show imports if no specific API impact was found
            print(f"\n   Import Locations:")
            for loc in r["import_locations"][:5]:
                print(f"     • {loc['file']}:{loc['line']}")

        print(f"\n   Upgrade Command:")
        print(f"     {r['upgrade_command']}")

    print("\n" + "=" * 70)
    print(f"Total: {len(results)} package(s) analyzed")
    print("=" * 70)


def positive_int(value: str) -> int:
    """Validate that the argument is a positive integer."""
    try:
        int_value = int(value)
        if int_value <= 0:
            raise argparse.ArgumentTypeError(f"{value} must be a positive integer (got {int_value})")
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be a valid integer")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze package upgrades using Tavily Research API (advisory only - never auto-upgrades)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --path /my/project
  %(prog)s --packages flask numpy --json
  %(prog)s --output report.json --max-wait 300
        """,
    )
    parser.add_argument(
        "--path", "-p",
        help="Project path (default: current directory)"
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        help="Specific packages to analyze (default: all outdated)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON to stdout"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save JSON results to file"
    )
    parser.add_argument(
        "--poll-interval",
        type=positive_int,
        default=5,
        help="Seconds between research status checks (must be positive, default: 5)"
    )
    parser.add_argument(
        "--max-wait",
        type=positive_int,
        default=180,
        help="Maximum seconds to wait for research per package (must be positive, default: 180)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize Tavily client
    api_key = os.environ.get("TAVILY_API_KEY")
    tavily_client = None

    if api_key and TavilyClient:
        try:
            tavily_client = TavilyClient(api_key=api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily client: {e}")
            logger.warning("Risk analysis will be skipped. Continuing without Tavily.")
            tavily_client = None
    else:
        logger.warning("Tavily not available. Risk analysis will be skipped.")
        logger.warning("Set TAVILY_API_KEY and install tavily-python for full analysis.")

    results = analyze_packages(
        project_path=args.path,
        specific_packages=args.packages,
        tavily_client=tavily_client,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
    )

    # Output handling
    if args.output:
        output_path = Path(args.output)
        try:
            output_path.write_text(json.dumps(results, indent=2))
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            return 1

    if args.json:
        print(json.dumps(results, indent=2))
    elif not args.output:
        # Only print human-readable if not saving to file and not JSON mode
        print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
