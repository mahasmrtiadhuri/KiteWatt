# KiteWatt

**Version 2.1:** Added an honest competitive audit against direct code and measured in-process caching. See [COMPETITIVE_AUDIT.md](docs/COMPETITIVE_AUDIT.md).

**Version 2.0:** A real local-model evaluation path is now available through the optional Ollama runtime. The default offline benchmark uses a Python reference; the shipped example is not a verified real-LLM result. See [MODEL_BENCHMARK.md](docs/MODEL_BENCHMARK.md).

**More intelligence. Less compute.**


**An AI efficiency lab that attempts to turn repeated model behavior into smaller, verified programs—and knows when not to use them.**

A standalone, local-first research prototype (no paid API keys required). It includes a small neural teacher, a bounded program compiler, validation-first routing, a cost gate, reproducible benchmark reports, and an interactive local dashboard.


> **Research status:** Functional prototype, **not AGI** and **not a novel state-of-the-art architecture**. Small-program synthesis, caching, and cost-aware routing all have established precedents. KiteWatt is intended as a rigorous playground for studying when skill compilation improves total cost.

## New in v2: benchmark an actual local model

No API keys or paid inference service are needed. First run the offline evaluation:

```bash
kitewatt model-bench --backend reference --task slug --output runs/offline
```

To evaluate a **real model running on your own machine**, install Ollama separately, download an appropriate open-weight model (subject to its license), then run:

```bash
ollama pull llama3.2:1b
kitewatt model-bench --backend ollama --model llama3.2:1b \
  --task slug --test-count 24 --output runs/ollama-slug
```

This uses `+"`http://127.0.0.1:11434`"+` only. Nothing is sent to paid AI APIs. If the model is unavailable, the command fails rather than substituting artificial results. Python reference mode is for validating our pipeline, **not** for claiming actual LLM savings.

The model benchmark compares a teacher, a compiled rule, full router overhead, an analytical exact-cache call-count baseline, and a handwritten correctness oracle. Deployment requires passing both held-out accuracy and amortized latency tests. Actual joules are reported only when a compatible hardware counter is accessible. Read the [methodology](docs/MODEL_BENCHMARK.md) before drawing conclusions.

## Competitive evaluation: is KiteWatt actually better?

Do **not** judge the project by beating a needlessly expensive model on a trivial
transformation. For the present tasks, a cache and a handwritten function are
stronger baselines. The new audit measures all of them on held-out inputs:

```bash
kitewatt competitive-audit --task slug --output runs/competitive/slug
```

The local reference experiment did **not** show a competitive win; see
[the audit and next-step decision](docs/COMPETITIVE_AUDIT.md). External
libraries and live production models have not yet been benchmarked head-to-head.

## Quick start

Requires Python 3.10+ and enough memory for a small CPU PyTorch model. No API key, paid model, or external inference service.

```bash
git clone https://github.com/mahasmrtiadhuri/KiteWatt.git
cd KiteWatt
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest
kitewatt serve
```

Open **http://127.0.0.1:8765** to use the local dashboard.

![KiteWatt local research dashboard](docs/dashboard-preview.webp)
 The server binds to loopback only. You can run everything from the terminal too:

```bash
# Learn three text transformations from examples and verify on sealed test inputs
kitewatt text-lab --seed 7 --output runs/text-demo

# Train a tiny Transformer on a finite numerical task and attempt compilation
kitewatt demo --operator max --steps 300 --requests 512 --output runs/neural-demo

# Query the compiled numerical rule once it exists
kitewatt query --skill runs/neural-demo/skill.json --left 2 --right 7
```

For a quick first test, use `--steps 60` instead of 300. A shorter run may not train a sufficiently accurate teacher to compile a skill; reporting that failure is part of the experiment.

## What actually happens

```text
Small neural model or reference teacher
       |  example observations
       v
Bounded program search + informative queries
       |  candidate program
       v
Separate validation examples
       |  acceptable accuracy?
       v
Cost gate: does compilation repay its overhead?
       |  yes                         |  no
       v                              v
Run a certified small skill         Use original solver
       |                              |
       +------- report held-out results and compute cost -----+
```

The **numerical path** trains a small Transformer on every input in a finite domain. It compiles a skill from a subset of the teacher's answers, validates on separate examples, and tests a reserved split. The student test split was **not unseen by the teacher**. A full-domain cache and handwritten Python are included as stronger baselines.

The **text path** uses a *handwritten Python reference teacher*, not a neural model. It synthesizes bounded programs for URL slugs, email domains, and URL path segments. An active-learning loop chooses input examples that distinguish candidate programs; then the system checks separate validation examples and a final held-out split. This is an educational program-synthesis experiment, not a claim about LLM text understanding.

## What makes the project worth exploring?

The research question is **not** whether a compiled Python function is faster than a neural network on a trivial problem. That's expected. The interesting system-level question is whether a learner can automatically identify which skills are reusable, validate them, monitor for failure, and account for the compute consumed to acquire those skills.

In the text lab, MOLT **does not deploy** correct compiled skills when they are slower than the handwritten reference. The system reports the failure and leaves the cheaper path in place. This is intentional.

## Results and limitations

Results vary by CPU, PyTorch version, operating system, and background load. One measured local neural demo (max of two small integers, 256 test requests, seed 7) produced approximately:

| Approach | Median request latency | Correctness |
|---|---:|---:|
| Tiny Transformer | 0.226 ms | 100% |
| MOLT compiled router | 0.00067 ms | 100% |
| Full-domain lookup cache | 0.00009 ms | 100% |
| Handwritten Python | 0.00012 ms | 100% |

The cache and handwritten baseline are faster than MOLT. **We measured no reliable energy reduction** on this host; the power-counter result was unavailable. These are narrow results on a toy task, not evidence of broad AI inference efficiency. See `examples/neural_max_report.json` and `docs/EVALUATION.md` for the full methodology.

Energy, latency, cost, and token usage are different metrics. A 300× lower CPU latency on one toy call does not mean 300× lower real-world electricity usage.

## Repository map

```text
kitewatt/
  model.py               Tiny decoder-only Transformer
  molt/
    benchmark.py         Neural teacher + four-way comparison
    language.py          Bounded integer DSL
    compiler.py          Validated integer skill compilation
    runtime.py           Fallback routing and independent spot checking
    text.py              Bounded string DSL + active query-by-committee
    lab.py               Disjoint train/validation/test text experiments
    gate.py              Accuracy, latency, and break-even deployment gate
    store.py             JSON skill persistence (no eval)
    ollama.py            Optional localhost-only model-teacher adapter
    web.py               Localhost-only dashboard server
    dashboard.html       Self-contained responsive web interface
    energy.py            Best-effort Linux RAPL power counter
    cli.py               Command line interface

tests/                    Automated regression and integration tests
examples/                 Real measured example reports
docs/EVALUATION.md        Fairness, leakage, limits and next experiments
.github/workflows/        CPU CI using free GitHub Actions allowances
```

## Security and privacy

- The local dashboard listens on `127.0.0.1` only; **do not expose it to the public internet**.
- Program specifications use an allowlisted DSL; they never contain arbitrary Python code or shell commands.
- Saved certificates re-check their stored validation examples on load, but a local file can still be forged by someone with filesystem access. Do not treat these as authenticated proofs.
- Query inputs are bounded, domain-checked, and fall back to a reference method when appropriate.
- Experiments make no model-provider network calls and do not send user inputs to a remote API.
- The energy meter is best effort and may include unrelated CPU activity; lack of a reading means energy is **not measured**.

## What would count as a real breakthrough?

Success on a useful, unseen **real-world reasoning workload**, reproducible improvements against **caching, model routing, distillation, and optimized inference**, comparable accuracy, measured actual energy rather than latency proxies, and an independently reproduced benefit *after including compilation costs*. We have **not achieved** that yet.

See [Evaluation](docs/EVALUATION.md) for stronger experiments and [Security](SECURITY.md) for system boundaries.

## Optional: use an actual locally hosted LLM as a teacher

The default lab is fully offline and deliberately uses a reference function, not a language model. To try skill compilation from a **real local model** instead, install [Ollama](https://ollama.com), download a suitable model, and run it locally:

```bash
ollama pull llama3.2:1b
kitewatt ollama-demo --model llama3.2:1b --task slug --output runs/local-model
```

This adapter sends requests only to `http://127.0.0.1:11434`, with no paid model APIs. It uses an independently implemented rule verifier on validation and held-out test samples. Local model responses may be inconsistent, the compiler may refuse to compile, and downloading/running weights requires disk space, RAM, and electricity. We have **unit-tested the adapter** but have not validated an actual Ollama run on this machine.
