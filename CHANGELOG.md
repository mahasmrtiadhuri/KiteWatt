# Changelog

## 2.0.0

- Added opt-in real local-Ollama model workflow benchmarking without paid APIs.
- Added held-out compiler evaluation against independent task oracles, plus time- and accuracy-aware routing gates.
- Captured Ollama-reported prompt, output token, and server duration telemetry when available.
- Added explicitly partitioned, size-bounded JSONL workload input for local testing.
- Compared against an analytical cold/warm exact-cache baseline and full routing overhead.
- Extended the local dashboard, CLI, CI smoke tests, and methodology documentation.
- Continued to label energy measurements unavailable when no hardware source is present.


## 1.0.0 — Standalone research lab

- Extracted MOLT into its own installable repository structure.
- Added bounded text-skill synthesis with disagreement-guided examples and sealed validation/test splits.
- Added a cost gate that can refuse a compiled skill when it is slower, inaccurate, or too expensive to amortize.
- Added a full-domain lookup-cache baseline alongside neural, compiled, and hand-written approaches.
- Added optional localhost Ollama model teaching and periodic drift checks.
- Added a self-contained local web dashboard with interactive experiments and compiled skill queries.
- Added skill revalidation on load, CLI tools, CI smoke tests, and benchmark methodology.

## Predecessor: NOVA v0.4 research module

- Tiny neural teacher, integer-skill compiler, finite-domain validation, fallback routing, JSON skill storage, best-effort power measurements.
