# Contributing

Keep changes focused and reproducible. For model or compiler changes, include a test that fails without the change, describe the task distribution and held-out split, and report accuracy alongside speed. Avoid benchmark claims that omit caching, reference code, or the cost of acquiring a skill.

Run `python -m pytest` and both CLI demos before requesting review. Commit small sample results as JSON, but never commit trained checkpoint weights, secrets, virtual environments, or machine-specific caches.
