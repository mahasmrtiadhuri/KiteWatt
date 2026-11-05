"""Reproducible competitive audit against simple application-level baselines.

All selection happens on validation inputs. The test split is never used to
choose a compiled skill or tune a cost threshold. In reference mode the teacher
is a Python function, NOT an LLM. The timings are wall-clock estimates, not
energy measurements or claims about external competing products.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter_ns
from typing import Callable

from .lab import generate, oracle
from .text import TASKS, TextRouter, induce


@dataclass(frozen=True)
class AuditConfig:
    task: str = "slug"
    seed: int = 7
    discovery: int = 48
    validation: int = 24
    test: int = 32
    repeat: int = 10
    rounds: int = 5
    expected_requests: int = 10_000
    output: str = "runs/competitive-audit"


def _validate(config: AuditConfig) -> None:
    if config.task not in TASKS:
        raise ValueError("Unknown text task")
    if (min(config.discovery, config.validation, config.test) < 8
            or config.discovery + config.validation + config.test > 500
            or not 2 <= config.repeat <= 100
            or not 2 <= config.rounds <= 15
            or not 100 <= config.expected_requests <= 10_000_000):
        raise ValueError("Audit configuration outside supported bounds")


def _trial(
    task: str, stream: list[str], truth: list[str],
    teacher: Callable[[str], str], program: Callable[[str], str] | None,
    rounds: int,
) -> dict:
    """Time competing strategies on an identical stream; cache starts cold each round."""
    strategies = ["teacher", "exact_cache", "handwritten_reference"]
    if program is not None:
        strategies.append("compiled_router")
    timings: dict[str, list[float]] = {name: [] for name in strategies}
    predictions: dict[str, list[str]] = {}
    cache_counts = []
    for round_number in range(rounds):
        # Rotate order to reduce fixed-position warm-cache/CPU artifacts.
        order = strategies[round_number % len(strategies):] + strategies[:round_number % len(strategies)]
        for name in order:
            calls = 0
            cache: dict[str, str] = {}
            if name == "compiled_router":
                assert program is not None
                # This is intentionally the full router cost, not a bare DSL operation.
                # The program is used only after a separate validation certificate.
                fn = lambda item: program(item)
            elif name == "teacher":
                fn = teacher
            elif name == "handwritten_reference":
                fn = lambda item: oracle(task, item)
            else:
                def cached(item: str) -> str:
                    nonlocal calls
                    if item not in cache:
                        cache[item] = teacher(item)
                        calls += 1
                    return cache[item]
                fn = cached
            start = perf_counter_ns()
            outputs = [fn(item) for item in stream]
            duration = (perf_counter_ns() - start) / len(stream) / 1e6
            timings[name].append(duration)
            if round_number == 0:
                predictions[name] = outputs
            if name == "exact_cache":
                cache_counts.append(calls)
    return {
        name: {
            "heldout_accuracy": round(sum(a == b for a, b in zip(predictions[name], truth)) / len(truth), 5),
            "median_ms_per_request": round(median(timings[name]), 8),
            "min_ms_per_request": round(min(timings[name]), 8),
            "max_ms_per_request": round(max(timings[name]), 8),
        }
        for name in strategies
    } | {
        "cache_diagnostics": {
            "cold_teacher_calls_per_round": cache_counts,
            "unique_requests": len(set(stream)),
            "total_requests": len(stream),
            "cold_hit_rate": round(1.0 - len(set(stream)) / len(stream), 5),
        }
    }


def run_competitive_audit(
    config: AuditConfig, *, teacher: Callable[[str], str] | None = None,
) -> dict:
    _validate(config)
    if teacher is None:
        teacher = lambda value: oracle(config.task, value)
    inputs = generate(config.task, config.discovery + config.validation + config.test, config.seed)
    discovery = inputs[:config.discovery]
    validation = inputs[config.discovery:config.discovery + config.validation]
    test = inputs[-config.test:]
    assert not set(discovery) & set(validation) and not set(discovery) & set(test) and not set(validation) & set(test)

    start = perf_counter_ns()
    compiled = induce(config.task, teacher, discovery, validation,
                      max_queries=min(18, config.discovery),
                      verifier=lambda value: oracle(config.task, value))
    compile_ms = (perf_counter_ns() - start) / 1e6
    skill = compiled.skill
    candidate_fn = None
    if skill is not None:
        router = TextRouter(lambda _, value: teacher(value))
        router.install(skill)
        candidate_fn = lambda value: router.predict(config.task, value)

    def sample_stream(items: list[str], offset: int) -> list[str]:
        values = items * config.repeat
        random.Random(config.seed + offset).shuffle(values)
        return values

    validation_stream = sample_stream(validation, 101)
    validation_result = _trial(config.task, validation_stream,
                               [oracle(config.task, x) for x in validation_stream],
                               teacher, candidate_fn, config.rounds)
    baseline_names = ("teacher", "exact_cache", "handwritten_reference")
    validation_best = min(validation_result[name]["median_ms_per_request"] for name in baseline_names)
    candidate_val = validation_result.get("compiled_router")
    projected_candidate_ms = (
        compile_ms / config.expected_requests + candidate_val["median_ms_per_request"]
        if candidate_val else None
    )
    deploy = bool(candidate_val and
                  candidate_val["heldout_accuracy"] == 1.0 and
                  candidate_val["heldout_accuracy"] >= validation_result["teacher"]["heldout_accuracy"] and
                  projected_candidate_ms is not None and
                  projected_candidate_ms < 0.8 * validation_best)

    # Test results are published whether or not the candidate was selected;
    # they are never used to change the selection above.
    test_stream = sample_stream(test, 202)
    test_result = _trial(config.task, test_stream,
                         [oracle(config.task, x) for x in test_stream],
                         teacher, candidate_fn, config.rounds)
    competitor = min(baseline_names, key=lambda name: test_result[name]["median_ms_per_request"])
    report = {
        "project": "KiteWatt",
        "type": "offline competitive audit; no LLM or physical energy measurement",
        "config": asdict(config),
        "partitions": {"discovery": len(discovery), "validation": len(validation), "test": len(test)},
        "compiler": {
            "status": compiled.status, "teacher_calls": compiled.teacher_calls,
            "programs_examined": compiled.programs_examined,
            "compile_ms": round(compile_ms, 5),
            "operations": list(skill.program.operations) if skill else None,
        },
        "validation": validation_result,
        "validation_decision": {
            "deploy": deploy,
            "best_baseline_ms": validation_best,
            "candidate_amortized_ms": round(projected_candidate_ms, 8) if projected_candidate_ms is not None else None,
            "required_relative_gain": 0.2,
            "reason": "passes accuracy and amortized latency gate" if deploy else
                      "candidate absent, less accurate, or not at least 20% faster than strongest simple baseline",
        },
        "heldout_test": test_result,
        "test_fastest_simple_baseline": competitor,
        "research_decision": (
            "candidate merited further independent testing" if deploy else
            "NO-GO: no demonstrated improvement over simple baselines on this workload"
        ),
        "limitations": [
            "The teacher is a Python reference unless an explicit custom callable is supplied; no live LLM was measured.",
            "The three task families, DSL operations, and correct reference implementations are known in advance.",
            "The test set is held out from compilation and validation, but drawn from the same synthetic generator.",
            "An in-process exact cache is measured; this is not a benchmark of GPTCache, LiteLLM or Redis products.",
            "The handwritten reference is an honest strong baseline for these simple tasks.",
            "Latency is wall-clock CPU time on this machine, not joules, total cost or production throughput.",
            "Tiny timings can vary dramatically across machines and repeated runs; no cross-system performance claim is made.",
            "The deployment decision is made using validation only; held-out test results do not modify the gate.",
            "The one-time compilation cost is amortized over the configured expected_requests, not observed future traffic.",
        ],
    }
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "competitive_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
