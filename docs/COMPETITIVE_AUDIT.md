# KiteWatt competitive audit (November 2026)

## Decision

**NO-GO for claiming a competitive advantage on the current three text transformations.**
The compiled router is correct in these local experiments, but an exact cache and
simple handwritten Python perform the same tasks more cheaply. This is a research
finding, not a product failure or an endorsement of deploying unnecessary routing.

The relevant question is not whether the compiled rule is faster than an
unnecessarily invoked neural model. It is whether its **full lifecycle cost**
beats the cheapest reliable, practical alternative for the same workload.

## What existing systems already cover

| Existing approach | What it solves | Implication for KiteWatt |
| --- | --- | --- |
| [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) | Reuses unchanged prompt-prefix model state, with reduced cached-input rates | The cost baseline must already include provider caching when available. |
| [LiteLLM](https://docs.litellm.ai/docs/) | Multi-provider routing, retries, budgets, spend tracking, caching | KiteWatt cannot claim novelty from routing or cost monitoring alone. |
| [GPTCache](https://github.com/zilliztech/GPTCache) | Reuse of prior responses, including semantic similarity matching | Exact/semantic cache baselines must be evaluated for repeat-heavy workloads. |
| [ReaComp (2026)](https://arxiv.org/abs/2605.05485) | Compiles model reasoning into reusable symbolic solvers | Skill compilation itself is not an original research claim. |
| Ordinary Python / application rules | Exact simple tasks without an LLM call | This is often the strongest baseline and should not be omitted. |

These are product-capability comparisons, **not direct measured benchmarks of
external products**. We have not installed or run the entire competitor stack.

## Reproducible local comparison

```bash
pip install -e '.[dev]'
python -m pytest
kitewatt competitive-audit --task slug --repeat 10 --rounds 7 \
  --expected-requests 10000 --output runs/competitive/slug
kitewatt competitive-audit --task email_domain --repeat 10 --rounds 7 \
  --expected-requests 10000 --output runs/competitive/email_domain
kitewatt competitive-audit --task url_segment --repeat 10 --rounds 7 \
  --expected-requests 10000 --output runs/competitive/url_segment
```

Windows PowerShell users should put a command on one line instead of using
Unix shell backslashes for continuation.

The benchmark uses 48 discovery inputs, 24 independent validation inputs,
and 32 **untouched** test inputs by default. Each test input is repeated ten
times, with an initially empty exact cache each measurement round. Timing is
measured for the *whole* request stream, including cache lookup or router
logic, not just for a bare compiled operation. Each approach sees the same
requests. The selection gate is fixed using only validation measurements and
requires at least 20% projected amortized latency advantage over the fastest
simple competitor, without any accuracy loss. The test partition cannot alter
this decision.

The compiler uses a predefined bounded DSL and an independent reference
oracle. This is a deliberately favorable environment for deterministic code.
The reference teacher is a Python function, **not an LLM**. Because the
benchmark is run without a paid or downloaded model, it cannot validate
real-model cost or power savings.

### One observed CPU run (seed 7)

| Task | Teacher median ms/request | Exact cache median | Handwritten Python median | KiteWatt router median | Selected? |
| --- | ---: | ---: | ---: | ---: | --- |
| slug | 0.001286 | 0.000195 | 0.001185 | 0.003568 | No |
| email_domain | 0.000158 | 0.000084 | 0.000158 | 0.002211 | No |
| url_segment | 0.000301 | 0.000104 | 0.000290 | 0.001893 | No |

All methods were correct on these synthetic test sets. These times are
microbenchmarks from one CPU environment. Nanosecond-scale interpreter overhead,
scheduling, warmup, and Python version can change the ranking. We make **no
energy-saving or cross-machine performance claim** from them. The check is
useful because the router fails to beat even ordinary alternatives.

Machine-readable audit results are stored under `examples/competitive_audit_seed7.json`.
Each local run emits `runs/competitive/<task>/competitive_report.json`.

## The real gap we would need to prove

A defensible differentiator would be **safe task-family discovery from actual
application traces**: after seeing examples of a costly recurring workflow,
the system proposes a compact reusable skill, checks it against an independent
verifier or formal specification, and deploys it only when it beats a realistic
cache, routing, and straightforward-program alternative **over its expected
lifetime**. Neither novelty nor commercial demand has been established.

### Pre-registered next experiment

1. Choose one expensive structured workflow where handwritten rules are not
   already sufficient; use genuinely independently labeled, permissioned data.
2. Fix the discovery, validation, and test splits in advance. Isolate tenants,
   private data, provider configuration, and any shared caches.
3. Benchmark direct model calls with the provider's caching enabled, an exact
   cache, a semantic cache, a strong manually engineered solution, and KiteWatt.
4. Measure task success, error severity, total tokens, warm/cold latency,
   fallback rate, compile/verification expense, and joules where instrumentation
   is available. Use matched hardware and independent workloads.
5. Require no measurable quality regression and a substantial cost advantage
   across multiple random seeds and workload shifts before releasing a claim.

### Stop / pivot criteria

- **Stop:** A few lines of normal code achieve the same outcome at lower cost.
- **Stop:** Cache or existing router performs equally well, or compilation cannot
  repay itself over realistic request volume.
- **Stop:** There is no independent way to check output correctness in the
  proposed task family.
- **Continue:** On a substantial, independently labeled workflow, KiteWatt
  reliably beats all relevant baselines under comparable conditions.

**Next decision:** retain KiteWatt as a research foundation, but do not expand
its general-purpose feature set until a compelling real-world workflow and
measurement methodology are identified.
