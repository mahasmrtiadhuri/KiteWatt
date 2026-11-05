"""Local-model, held-out workflow benchmark for bounded text skill compilation.

Model responses are observed, not used as ground truth. The independent task
oracle evaluates correctness, and the compiler cannot access the test split.
This is not a benchmark for open-ended reasoning or general algorithm invention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter, perf_counter_ns
from typing import Callable

from .energy import EnergyCounter
from .gate import decide
from .lab import generate, oracle
from .ollama import OllamaTeacher
from .text import TASKS, TextRouter, induce, supported


@dataclass(frozen=True)
class WorkloadConfig:
    task: str = "slug"
    backend: str = "reference"
    model: str = "llama3.2:1b"
    port: int = 11434
    seed: int = 7
    discovery: int = 48
    validation: int = 20
    test: int = 24
    max_queries: int = 12
    repeat: int = 3
    expected_requests: int = 10_000
    output: str = "runs/workload"
    dataset_file: str | None = None


def load_dataset(path: Path, task: str) -> tuple[list[str], list[str], list[str]]:
    """Require explicit, disjoint partitions; never silently split a supplied dataset."""
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("Dataset must be a JSONL file no larger than 1 MB")
    parts: dict[str, list[str]] = {"discovery": [], "validation": [], "test": []}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on dataset line {number}") from error
        if (not isinstance(item, dict) or set(item) != {"task", "split", "input"}
                or item["split"] not in parts or item["task"] not in TASKS
                or not isinstance(item["input"], str)):
            raise ValueError(f"Unexpected dataset schema on line {number}")
        if item["task"] == task:
            if not supported(task, item["input"]):
                raise ValueError(f"Unsupported input on dataset line {number}")
            parts[item["split"]].append(item["input"])
    all_inputs = [x for group in parts.values() for x in group]
    if len(all_inputs) != len(set(all_inputs)):
        raise ValueError("Dataset inputs must be disjoint and unique across splits")
    if min(map(len, parts.values())) < 8 or max(map(len, parts.values())) > 500:
        raise ValueError("Each task needs 8–500 examples per split")
    return parts["discovery"], parts["validation"], parts["test"]


def _time_calls(fn: Callable[[str], str], inputs: list[str]) -> tuple[list[str], list[float]]:
    outputs: list[str] = []
    times: list[float] = []
    for value in inputs:
        start = perf_counter_ns()
        outputs.append(fn(value))
        times.append((perf_counter_ns() - start) / 1e6)
    return outputs, times


def _timing(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot measure empty input")
    return {
        "mean_ms": round(mean(values), 6),
        "median_ms": round(median(values), 6),
        "p95_ms": round(sorted(values)[int(0.95 * (len(values) - 1))], 6),
    }


def _accuracy(outputs: list[str], expected: list[str]) -> float:
    return round(sum(a == b for a, b in zip(outputs, expected)) / len(expected), 5)


def run_workload(config: WorkloadConfig, *, teacher: Callable[[str], str] | None = None) -> dict:
    if config.task not in TASKS or config.backend not in {"reference", "ollama"}:
        raise ValueError("Unknown task or backend")
    if (not 8 <= config.discovery <= 500 or not 8 <= config.validation <= 500
            or not 8 <= config.test <= 500 or not 6 <= config.max_queries <= min(config.discovery, 64)
            or not 1 <= config.repeat <= 30 or not 1 <= config.expected_requests <= 10_000_000):
        raise ValueError("Benchmark configuration outside supported bounds")
    provider = None
    if teacher is None:
        if config.backend == "ollama":
            provider = OllamaTeacher(config.model, port=config.port)
            teacher = lambda text: provider(config.task, text)
        else:
            teacher = lambda text: oracle(config.task, text)
    calls = 0
    raw_teacher = teacher

    def observed_teacher(text: str) -> str:
        nonlocal calls
        result = raw_teacher(text)
        calls += 1
        return result
    if config.dataset_file is not None:
        discovery, validation, test = load_dataset(Path(config.dataset_file), config.task)
    else:
        values = generate(config.task, config.discovery + config.validation + config.test,
                          config.seed)
        discovery = values[:config.discovery]
        validation = values[config.discovery:config.discovery + config.validation]
        test = values[-config.test:]
    if set(discovery) & set(test) or set(validation) & set(test):
        raise AssertionError("Test input leakage")

    calls_before = calls
    started = perf_counter()
    compiled = induce(config.task, observed_teacher, discovery, validation,
                      max_queries=config.max_queries,
                      verifier=lambda text: oracle(config.task, text))
    compile_seconds = perf_counter() - started
    discovery_calls = calls - calls_before
    # The verifier is a separate deterministic reference. Use validation
    # examples for BOTH the quality/cost gate and router benchmarking.
    # The final test set must not influence model/skill selection.
    validation_truth = [oracle(config.task, value) for value in validation]
    validation_model, validation_model_ms = _time_calls(observed_teacher, validation)
    validation_model_quality = _accuracy(validation_model, validation_truth)

    compiled_quality: float | None = None
    compiled_latencies: dict | None = None
    gate_report: dict | None = None
    routed_latency: dict | None = None
    deployment = False
    router = TextRouter(lambda name, value: observed_teacher(value))
    if compiled.skill is not None:
        validation_compiled, validation_compiled_ms = _time_calls(
            compiled.skill.predict, validation * config.repeat
        )
        calibration_quality = _accuracy(validation_compiled[:len(validation)], validation_truth)
        gate = decide(reference_ms=max(mean(validation_model_ms), 0.000001),
                      compiled_ms=max(mean(validation_compiled_ms), 0.000001),
                      compile_seconds=compile_seconds, measured_accuracy=calibration_quality,
                      expected_requests=config.expected_requests)
        router.install(compiled.skill)
        routed_outputs, router_ms = _time_calls(
            lambda text: router.predict(config.task, text), validation * config.repeat
        )
        routed_latency = _timing(router_ms)
        routed_accuracy = _accuracy(routed_outputs[:len(validation)], validation_truth)
        route_gate = decide(reference_ms=max(mean(validation_model_ms), 0.000001),
                            compiled_ms=max(mean(router_ms), 0.000001),
                            compile_seconds=compile_seconds,
                            measured_accuracy=routed_accuracy,
                            expected_requests=config.expected_requests)
        deployment = (gate.deploy and route_gate.deploy and
                      calibration_quality >= validation_model_quality and
                      routed_accuracy >= validation_model_quality)
        gate_report = {**route_gate.to_dict(), "deploy": deployment,
                       "bare_compiled_gate": gate.to_dict(),
                       "calibration_accuracy": calibration_quality,
                       "calibration_model_accuracy": validation_model_quality,
                       "quality_check": "pass" if calibration_quality >= validation_model_quality else "below_teacher"}

    # A fully held-out final measurement happens AFTER deployment selection.
    meter = EnergyCounter.discover()
    power_before = meter.read() if meter else None
    outputs, model_ms = _time_calls(observed_teacher, test)
    power_after = meter.read() if meter else None
    model_joules = meter.delta_joules(power_before, power_after) if meter else None
    truth = [oracle(config.task, value) for value in test]
    model_quality = _accuracy(outputs, truth)
    if compiled.skill is not None:
        compiled_predictions, test_compiled_ms = _time_calls(
            compiled.skill.predict, test * config.repeat
        )
        compiled_quality = _accuracy(compiled_predictions[:len(test)], truth)
        compiled_latencies = _timing(test_compiled_ms)

    # Cold exact cache: the first occurrence of each unseen test input must
    # be answered by the model. Subsequent repeated requests are cache hits.
    # This is a logical call-count baseline, not an observed end-to-end timing.
    cold_calls = len(test)
    repeated_requests = len(test) * config.repeat
    model_usage = {
        "teacher_calls_during_induction": discovery_calls,
        "teacher_calls_during_gate_calibration": len(validation),
        "teacher_calls_during_test": len(test),
        "total_teacher_calls_observed": calls,
        "server_prompt_tokens_observed": getattr(provider, "prompt_tokens", None),
        "server_generated_tokens_observed": getattr(provider, "completion_tokens", None),
        "server_total_duration_seconds_observed": getattr(provider, "server_seconds", None),
    }
    report = {
        "project": "KiteWatt", "benchmark": "local text workflow; bounded compilation",
        "config": asdict(config),
        "partitions": {"discovery": len(discovery), "validation": len(validation), "test": len(test)},
        "compiler": {"status": compiled.status, "candidate_count": compiled.programs_examined,
                     "teacher_queries": compiled.teacher_calls,
                     "validation_count": compiled.skill.validation_count if compiled.skill else len(validation),
                     "compile_seconds": round(compile_seconds, 5),
                     "operations": list(compiled.skill.program.operations) if compiled.skill else None},
        "model_reference": {"heldout_accuracy": model_quality, **_timing(model_ms),
                            "joules_unisolated_cpu_package": model_joules, **model_usage},
        "compiled_skill": {"heldout_accuracy": compiled_quality, "latency": compiled_latencies,
                           "gate": gate_report, "deployed": deployment,
                           "candidate_router_latency": routed_latency,
                           "candidate_router_paths": dict(router.counts)},
        "exact_cache": {"cold_requests": repeated_requests, "cold_model_calls": cold_calls,
                        "warm_cache_hits": repeated_requests - cold_calls,
                        "note": "Analytic call-count baseline; no cache speed or energy measured."},
        "comparison": {"handwritten_reference": "Independent reference function available for these predefined tasks",
                       "heldout_split_unseen_during_induction_and_gate": True,
                       "raw_model_accuracy_required": "Comparison uses independent task-oracle labels"},
        "limitations": [
            "This benchmark covers three predefined text transformations, not open-ended LLM reasoning.",
            "Model accuracy is measured on held-out inputs; the compiled skill is checked against independent reference labels.",
            "Teacher calls to create the skill and validation time may outweigh all subsequent savings.",
            "The exact-cache result is a logical call-count comparison, not measured runtime.",
            "CPU RAPL, if available, covers an unisolated package; it is not attributable total system or GPU energy.",
            "The reference backend uses Python functions, not an actual language model.",
            "The deployment gate uses the validation split; the final test split is measured only after selection.",
            "The candidate router is benchmarked but not deployed when either accuracy or cost checks fail.",
            "No benchmark result implies frontier-model parity or newly discovered algorithms.",
        ],
    }
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "workload_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if compiled.skill:
        (output / "skill.json").write_text(json.dumps(compiled.skill.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report
