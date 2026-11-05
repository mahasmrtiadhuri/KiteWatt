# Evaluation notes

## The three separate questions

1. **Can we compile?** Given a bounded set of input/output examples, identify a program in our restricted language and validate it independently.
2. **Does it generalize?** Test a sealed input partition that neither candidate selection nor validation can see. The numeric *teacher* has seen the full domain; the numerical test is held out only from the compiler. Text samples are drawn from the same limited synthetic generator.
3. **Is it cheaper?** Compare against a neural reference, full-domain caching, and hand-written code where relevant. Include program search, oracle calls, deployment checks, memory overhead, and ongoing monitoring when deciding lifetime benefit.

## Baselines, correctly interpreted

- **Neural teacher**: 30,144-parameter Transformer trained from scratch on a complete finite integer function. Not a frontier model or optimized system.
- **MOLT**: compiled bounded DSL rule selected from candidate programs and verified on separate inputs. It is usually faster than the neural model because it performs much less work.
- **Lookup cache**: strong baseline when the domain is finite and every input can be enumerated. It is often faster than MOLT and can be simpler.
- **Hand-written Python**: strongest reasonable baseline when an exact function is already known. MOLT should not claim to have improved on it when it has not.
- **Text teacher**: original Python function (not an LLM), included to isolate program synthesis and cost gating. No semantic reasoning is implied.

## Metrics we report

- Accuracy on held-out compiler tests.
- Median and p95 request latency, and time spent compiling.
- Teacher query counts for discovery **and** separate validation.
- An estimated break-even request count (compilation time divided by mean inference time saved); this does not include neural teacher training, hardware lifetime costs, or energy.
- CPU package joules only if a readable Linux RAPL counter exists. A `null` means **unavailable**, not zero energy.
- Whether the cost gate accepted or refused a compiled skill.

## Additional caveats

- Most microsecond-scale timings vary with CPU frequency, process scheduling, and warm-up. Reproduce on multiple hosts and runs before making performance claims.
- The compiler searches a small built-in DSL. It cannot invent arbitrary algorithms, prove equality for all possible strings, or find operations outside that DSL.
- Text validation examples are generated independently but from the same family, so 100% on that distribution is **not** robust out-of-domain correctness.
- The deployment gate uses latency to decide whether a skill is worth considering; it does not measure power or prove that actual energy use is lower.
- A serialized skill is not a cryptographically authenticated proof or an executable safety sandbox.
- Do not compare raw speedups between different hardware, model sizes, or request batches.

## Roadmap to credible AI research

1. Replace toy teachers with **locally hosted, properly licensed small language models** on real developer workflows; preserve a no-paid-API option.
2. Define held-out tasks, new program families, and distribution shifts; avoid training and benchmark leakage.
3. Compare with retrieval cache, semantic cache, model cascades, classical program synthesis, and task-specific compiled programs under a matched quality constraint.
4. Measure full-system energy (e.g., hardware power meter or correctly attributed RAPL samples) and amortize training/compilation plus monitoring.
5. Conduct ablations: no active querying, no validation, no cost gate, no fallback, and different DSL complexities.
6. Obtain external replication before making claims of a new AI efficiency breakthrough.
