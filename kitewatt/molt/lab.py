"""Reproducible text-task lab with sealed test sets and transparent baselines."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import re
from statistics import mean, median
from time import perf_counter, perf_counter_ns
from urllib.parse import urlsplit

from .text import TASKS, ActiveResult, TextRouter, TextSkill, induce
from .gate import decide

WORDS = (
    "Nova", "Systems", "Deep", "Learning", "Memory", "Power", "Energy",
    "Quantum", "Compiler", "Research", "Signal", "Physics", "Graphs",
    "Language", "Reasoning", "Future", "Robotics", "Vision", "Tools",
    "Discovery", "Practical", "Reusable", "Adaptive", "Science",
)
DOMAINS = ("Example.com", "Research.io", "Nova.DEV", "OpenSource.org", "Systems.ai")


def oracle(task: str, text: str) -> str:
    """A deliberately inexpensive reference function, NOT an LLM teacher."""
    if task == "slug":
        return re.sub(r"\s+", "-", text.strip().lower())
    if task == "email_domain":
        return text.rsplit("@", 1)[-1].lower()
    if task == "url_segment":
        return urlsplit(text).path.rstrip("/").split("/")[-1].lower()
    raise ValueError("Unsupported task")


def generate(task: str, count: int, seed: int) -> list[str]:
    """Generate unique, in-domain examples. No network or external dataset."""
    if task not in TASKS or not 1 <= count <= 500:
        raise ValueError("Invalid task or size")
    rng = random.Random(seed)
    values: set[str] = set()
    for _ in range(count * 50):
        if len(values) == count:
            break
        if task == "slug":
            words = rng.sample(WORDS, k=rng.randint(2, 5))
            title = rng.choice((" ", "  ", "   ", "\t")).join(
                w.upper() if rng.random() < 0.28 else w.lower() if rng.random() < 0.4 else w
                for w in words
            )
            candidate = rng.choice(("", " ", "  ")) + title + rng.choice(("", "  ", " "))
        elif task == "email_domain":
            user = rng.choice(WORDS).lower() + str(rng.randrange(10_000))
            host = rng.choice(DOMAINS)
            host = host.lower() if rng.random() < 0.4 else host.upper() if rng.random() < 0.3 else host
            candidate = f"{user}@{host}"
        else:
            part = rng.choice(WORDS) + rng.choice(("", "-" + rng.choice(WORDS)))
            host = rng.choice(DOMAINS).lower()
            candidate = (f"https://{host}/{rng.choice(WORDS)}/{part}"
                         + rng.choice(("", "?src=email", "?ref=summer", "/")))
        values.add(candidate)
    if len(values) != count:
        raise RuntimeError("Could not produce enough unique input examples")
    result = sorted(values)
    rng.shuffle(result)
    return result


@dataclass(frozen=True)
class LabConfig:
    seed: int = 7
    discovery: int = 60
    validation: int = 24
    test: int = 40
    max_queries: int = 18
    repeat: int = 10
    output: str = "runs/text-lab"


def _measure(fn, points: list[str]) -> dict:
    timings = []
    predictions = []
    for text in points:
        start = perf_counter_ns()
        predictions.append(fn(text))
        timings.append(perf_counter_ns() - start)
    sorted_times = sorted(timings)
    return {
        "median_ms": round(median(timings) / 1e6, 7),
        "mean_ms": round(mean(timings) / 1e6, 7),
        "p95_ms": round(sorted_times[int(0.95 * (len(sorted_times) - 1))] / 1e6, 7),
        "predictions": predictions,
    }


def run_text_lab(config: LabConfig) -> dict:
    if not 8 <= config.discovery <= 300 or not 8 <= config.validation <= 100 or not 8 <= config.test <= 100:
        raise ValueError("Sample sizes outside supported bounds")
    if not 6 <= config.max_queries <= min(config.discovery, 64) or not 1 <= config.repeat <= 100:
        raise ValueError("Invalid query or repetition budget")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    router = TextRouter(oracle)
    report = {"benchmark": "MOLT bounded string-skill synthesis", "config": asdict(config), "tasks": {}}
    for i, task in enumerate(TASKS):
        inputs = generate(task, config.discovery + config.validation + config.test, config.seed + i * 100)
        discovery = inputs[:config.discovery]
        validation = inputs[config.discovery:config.discovery + config.validation]
        test = inputs[-config.test:]
        if set(discovery) & set(test) or set(validation) & set(test):
            raise AssertionError("Split leakage")
        compile_started = perf_counter()
        result: ActiveResult = induce(task, lambda text: oracle(task, text), discovery, validation,
                                      max_queries=config.max_queries)
        elapsed = perf_counter() - compile_started
        if result.skill:
            # No arbitrary Python code is stored or executed by the skill book.
            (output / f"{task}.skill.json").write_text(
                json.dumps(result.skill.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        sample = test * config.repeat
        baseline = _measure(lambda x: oracle(task, x), sample)
        compiled = _measure(lambda x: result.skill.predict(x), sample) if result.skill else None
        quality = (sum(a == b for a, b in zip(compiled["predictions"], baseline["predictions"])) / len(sample)
                   if compiled else 0.0)
        gate = (decide(reference_ms=baseline["median_ms"], compiled_ms=compiled["median_ms"],
                       compile_seconds=elapsed, measured_accuracy=quality)
                if compiled else None)
        if gate and gate.deploy:
            router.install(result.skill)
        routed = _measure(lambda x: router.predict(task, x), sample)
        correctness = sum(a == b for a, b in zip(baseline["predictions"], routed["predictions"])) / len(sample)
        entry = {
            "status": result.status,
            "program": list(result.skill.program.operations) if result.skill else None,
            "teacher_calls_for_discovery": result.teacher_calls,
            "teacher_calls_for_validation": len(validation) if result.skill else len(validation),
            "candidate_programs": result.programs_examined,
            "candidates_remaining": result.survivors,
            "validation_accuracy": result.skill.verified_accuracy if result.skill else None,
            "heldout_test_accuracy": round(correctness, 5),
            "compilation_seconds": round(elapsed, 5),
            "reference_median_ms": baseline["median_ms"],
            "compiled_median_ms": compiled["median_ms"] if compiled else None,
            "router_median_ms": routed["median_ms"],
            "cost_gate": gate.to_dict() if gate else None,
            "routing": {"compiled": len(sample) if gate and gate.deploy else 0,
                        "fallback": 0 if gate and gate.deploy else len(sample)},
            "discovery_trace": list(result.trace),
            "test_count": len(test),
            "test_inputs_used_during_compilation": 0,
        }
        report["tasks"][task] = entry
    report["measurement_notes"] = [
        "The teacher here is a handwritten Python reference, NOT a neural model or a commercial API.",
        "Held-out test inputs were not available to the compiler; the synthetic input distribution remains narrow.",
        "Correctness compares with the reference teacher, and compilation can still fail on other inputs.",
        "The compiled DSL can be slower than handwritten Python; the cost gate then selects the reference.",
        "No electricity or training-energy savings are implied by this benchmark.",
        "Timing numbers are platform dependent and lack confidence intervals.",
        "Programs are bounded DSL specifications, never generated executable Python code.",
    ]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
