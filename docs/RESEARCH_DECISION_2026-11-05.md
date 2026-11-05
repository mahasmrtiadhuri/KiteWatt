# KiteWatt research decision — 2026-11-05

**Status: research hypothesis only. No new product feature authorized by this document.**

## Executive decision

**NO-GO:** general-purpose caching, gateway routing, prompt compression, generic trace-to-program compilation, or a claim of a novel energy-efficient AI architecture.

**CONDITIONAL GO:** a *verified AI-call elimination auditor* for high-volume, narrowly scoped structured workflows. The auditor identifies candidate model calls that could be removed; it must independently verify replacements, compare against strong baselines, and show lifecycle savings. Begin in **read-only/shadow mode**; do not automatically modify production behavior.

The proposal is **not established as original**. The research contribution or customer advantage would need to be proven.

## Competitive evidence

| Existing approach | Documented capability | Consequence |
| --- | --- | --- |
| [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) | Reuses cached prompt prefixes and discounts repeated input | Count provider cache pricing as the baseline |
| [LiteLLM caching](https://docs.litellm.ai/docs/proxy/caching) and [routing](https://docs.litellm.ai/docs/routing) | Exact/semantic caching, routing, provider failover | Do not build another generic gateway |
| [Portkey](https://portkey.ai/features/ai-gateway) | Multi-provider routing, caching, monitoring | Cost reporting alone is not a differentiator |
| [ReaComp](https://arxiv.org/abs/2605.05485) | Compiles model reasoning traces into standalone symbolic solvers | Compiling reasoning is not novel by itself |
| [Microsoft AWO](https://www.microsoft.com/en-us/research/publication/optimizing-agentic-workflows-using-meta-tools/) | Mines repetitive agent traces into deterministic meta-tools | Workflow call elimination has research precedent |
| [Papaya](https://papaya.fyi/docs/how-it-works) | Mines real agent traces for reliability/cost improvements and recommends fixes | Even trace-driven optimization is competitive |
| [FlowCompile](https://arxiv.org/abs/2605.13647) | Compile-time design-space exploration of agent workflows | Global agent workflow compilation is an active field |

These sources establish competitive overlap, **not** measured head-to-head performance against KiteWatt.

## KiteWatt's existing evidence

See [COMPETITIVE_AUDIT.md](COMPETITIVE_AUDIT.md). In one local CPU audit of slug, email-domain, and URL-segment tasks, a conventional exact cache outperformed KiteWatt's compiled router, with all methods correct on narrow synthetic inputs. That result rules out marketing those three tasks as an efficiency breakthrough. The comparison used a Python teacher, not real model inference or direct joule readings.

## The only problem worth testing next

For an AI application that already uses a competent gateway and caching, can KiteWatt **detect a recurring costly deterministic subtask**, propose a smaller correct mechanism, and prove that removing the model call is **cheaper per successful task** over its operating lifetime?

The target is an *application-level verification layer*, not a model or GPU accelerator. The first candidate customer is a developer team processing stable-schema, repetitive structured requests at meaningful volume; this segment's willingness to pay is **not validated**.

## Minimum experiment and acceptance rules

1. Obtain 500–1,000 permissioned, independently labeled examples from one real workflow, or use a public/open licensed equivalent for the first technical pilot. Fix discovery/validation/test partitions before tuning, with at least one temporally shifted or altered-format test set.
2. Compare: **(A)** optimized direct model including provider prompt caching; **(B)** exact and semantic caching where safe; **(C)** strong hand-engineered deterministic rules; **(D)** low-cost model routing; **(E)** KiteWatt. Existing baselines must be configured competently.
3. Optimize **total cost per correctly completed request** including discovery, validation, monitoring, routing, fallback, and maintenance; report error severity, accuracy with confidence intervals, p50/p95 latency, request distribution, and inference-call savings. Only claim energy savings when directly measured with qualified hardware.
4. Run exclusively in shadow/read-only mode first. Reject deployment if any critical error, if baseline accuracy is not maintained within an agreed non-inferiority margin, or if incremental cost savings are non-positive after amortization.
5. Provisional go/no-go: demonstrate **at least 20% lower total cost per successful request** versus the strongest practical baseline on a held-out task family across multiple repeats and a shifted workload. This is an **aspirational threshold, not a predicted result**.
6. Validate business interest separately: interview at least 10 teams with meaningful AI bills; seek 2 organizations willing to provide consented traces or run a local pilot before building a hosted product.

## Next decision

**Do not start KiteWatt v3 features yet.** Find a credible pilot workflow, independently labeled data, and proof that present-day tools leave meaningful waste. If that evidence is absent, retain the repository as an open-source evaluation project rather than claim a startup opportunity.

All initial research and local software can be done without new paid API services. Production model tests or energy measurements may require customer access or suitable hardware, so a guaranteed $0 full commercial validation is not realistic.
