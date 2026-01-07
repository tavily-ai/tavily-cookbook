---
name: package-manager
description: "Analyze package upgrades using Tavily Research API. Returns risk level, breaking changes, code impact, and upgrade commands. Advisory only - never auto-upgrades."
---

# Package Manager Skill

Analyzes package dependencies using Tavily Research API and provides structured upgrade recommendations. **This skill is advisory only** - it never upgrades packages automatically. The user decides whether to run the upgrade commands.

## What This Skill Does

1. Detects outdated packages in a project
2. Researches each upgrade via Tavily Research API
3. Returns structured analysis for each package:
   - **Risk Level**: LOW / MEDIUM / HIGH
   - **Breaking Changes**: What APIs changed and how to migrate
   - **Code Impact**: Files in the codebase using affected APIs
   - **Upgrade Command**: The exact command to run

## How Claude Should Use This Skill

### User asks about outdated packages or upgrades

Run the analysis script:

```bash
python3 .claude/skills/package-manager/scripts/analyze_upgrades.py --path <project_path>
```

For specific packages only:
```bash
python3 .claude/skills/package-manager/scripts/analyze_upgrades.py --path <project_path> --packages flask numpy
```

Save results to a file:
```bash
python3 .claude/skills/package-manager/scripts/analyze_upgrades.py --path <project_path> --output report.json
```

With custom timeouts (for slow networks or large packages):
```bash
python3 .claude/skills/package-manager/scripts/analyze_upgrades.py --path <project_path> --poll-interval 10 --max-wait 300
```

### After running, present results as a table:

| Package | Version | Risk | Breaking Changes | Command |
|---------|---------|------|------------------|---------|
| flask | 2.0.0 → 3.1.2 | MEDIUM | `before_first_request` removed | `pip install Flask==3.1.2` |
| numpy | 1.24.0 → 2.0.2 | HIGH | `np.string_`, `np.unicode_` removed | `pip install numpy==2.0.2` |

### For HIGH risk packages, show details:

```
numpy 1.24.0 → 2.0.2 (HIGH RISK)

Breaking Changes:
  - np.string_ removed → use np.bytes_
  - np.unicode_ removed → use np.str_
  - np.mat deprecated → use np.array

Code Impact:
  - app/data_processor.py:15 uses np.string_
  - app/data_processor.py:18 uses np.unicode_

Upgrade Command:
  pip install numpy==2.0.2
```

**Let the user decide whether to run the upgrade command.**

## Output Schema

The Tavily Research API returns structured data:

```json
{
  "package": "flask",
  "current_version": "2.0.0",
  "latest_version": "3.1.2",
  "risk_level": "MEDIUM",
  "risk_explanation": "Breaking API changes require code updates",
  "breaking_changes": [
    {
      "affected_api": "before_first_request",
      "change": "Decorator removed in Flask 3.0",
      "migration": "Use @app.before_request with a flag"
    }
  ],
  "deprecated_apis": ["flask.ext.*"],
  "upgrade_command": "pip install Flask==3.1.2"
}
```

## Environment Variables

```bash
export TAVILY_API_KEY="your-tavily-key"  # Required
```

## Supported Package Managers

| Manager | Detection |
|---------|-----------|
| pip | `requirements.txt`, `pyproject.toml`, `setup.py`, `Pipfile` |
| npm | `package.json` (no lockfile or `package-lock.json`) |
| yarn | `yarn.lock` |
| pnpm | `pnpm-lock.yaml` |
