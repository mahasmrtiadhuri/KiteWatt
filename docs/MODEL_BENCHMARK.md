# KiteWatt v2 — Evaluating a local language model

**Scope:** Bounded text transformations (slug, email-domain, URL path segment). This is not general reasoning, autonomous program invention, or production-quality model optimization. The available program vocabulary is fixed before the experiment.

## Running locally

Install Python 3.10+ and PyTorch, then `pip install -e '.[dev]'`. To confirm the benchmark works without a model:

```bash
kitewatt model-bench --backend reference --task slug --output runs/offline
```

`reference` is a handwritten Python function, **not** an AI model. It is useful for correctness and overhead checks and often beats compiled programs.

To use a real open-weight model, install [Ollama](https://ollama.com/) locally and pull a model you are permitted to use (may require multiple GB of disk and sufficient RAM):

```bash
ollama pull llama3.2:1b
kitewatt model-bench --backend ollama --model llama3.2:1b --task email_domain --output runs/model
```

The benchmark uses additional validation model calls to calibrate the latency and reliability gate. Only after the gate is set does it evaluate the untouched test split.

The HTTP client only connects to loopback (`127.0.0.1`) and asks Ollama's `/api/generate` for deterministic short responses. Model downloads may require internet access, but the running benchmark does not call paid cloud APIs. No online account or server deployment is required. **Your computer still uses electricity.**

The CLI emits `workload_report.json` and, if a valid skill is found, `skill.json`. These files are created in the ignored `runs/` directory and should be reviewed before sharing: custom datasets or language-model responses may contain private information.

### Custom workload inputs

Use a local UTF-8 JSONL file of explicitly partitioned inputs. Each row has exactly these fields:

```json
{"task":"slug","split":"discovery","input":"  Deep  Research  "}
```

Allowed splits: `discovery`, `validation`, `test`. Input strings must be unique *across all splits*, within the task's declared domain. Include 8–500 examples for each split for the chosen task. The benchmark does **not** infer splits from ordering, and the compiler receives none of the `test` inputs or their labels. Run:

```bash
kitewatt model-bench --backend ollama --model llama3.2:1b --task slug --dataset-file my_workload.jsonl
```

**Limitations:** This JSONL loader supports the three declared tasks only, not arbitrary natural-language tasks. It deliberately refuses unknown task schemas rather than treating model output as ground truth.

## What the numbers mean

- **Held-out accuracy:** compare model output and compiled skill output separately to independent, fixed Python task oracles. All predictions are checked on test inputs not used for program search, calibration, or deployment gating.
- **Measured time:** model's observed per-request wall time, program's observed per-request wall time, and the full compiled router's observed per-request time. Timings depend heavily on hardware, contention, warmup, and batch size.
- **Model usage:** the adapter records server-reported prompt-token and generated-token counts and total-duration nanoseconds when Ollama supplies them. Counts cover all calls made for discovery and testing; they do not imply price or energy.
- **Gate:** a skill must match independent validation labels, be at least as accurate as its reference model on calibration, save at least 10% observed latency, and repay induction overhead within the expected number of future requests. The full router must also be faster; an individually fast program is not enough.
- **Exact cache:** a *call-count projection* for repeated requests. A cold cache must call the model on each distinct first request. Later exact repetitions can hit the cache. The cache's latency and energy are not measured in this report, and results are not a general-purpose semantic cache.
- **Energy:** optional Linux RAPL reports *unisolated CPU-package* joules for model calls, not model-only energy, GPU energy, full lifecycle electricity, or a comparable per-query joule estimate. `null` means **unavailable**, not zero.
- **Up-front cost:** includes discovery model calls and candidate search, but not development time, downloading model weights, or future monitoring. A break-even estimate does not prove positive lifetime energy savings.

## What to compare before claiming a breakthrough

1. An optimized, properly configured local LLM, with both cold and warm starts.
2. A handwritten reference for tasks with a known exact algorithm (often best).
3. Exact caching, and semantic caching only when correctness can be verified.
4. Compiled skill latency **including** routing and verification costs.
5. Multiple data distributions, multiple machines, repeat runs, and robust confidence intervals.

No real Ollama model weights are bundled. The project's automated tests use a simulated local API response to validate the integration; they do not establish performance on any actual model. No claims of AGI, algorithmic novelty, or measured electricity savings are made by this release.
