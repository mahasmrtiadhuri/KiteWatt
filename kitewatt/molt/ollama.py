"""Optional localhost-only model teacher for the text compiler.

No model weights are included, and no paid or external API is used. Users must
run an Ollama-compatible local model separately if they enable this adapter.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .lab import generate, oracle
from .text import TASKS, induce


PROMPTS = {
    "slug": "Turn the input into a lowercase URL slug. Trim spaces and replace whitespace with dashes.",
    "email_domain": "Return only the lowercase domain after the @ symbol in this email address.",
    "url_segment": "Return only the lowercase last nonempty path segment of this URL. Ignore query parameters.",
}


class OllamaTeacher:
    def __init__(self, model: str, *, port: int = 11434, timeout: float = 90.0) -> None:
        if not model or not isinstance(model, str) or len(model) > 80 or any(c in model for c in "\r\n\x00"):
            raise ValueError("Invalid local model name")
        if not 1 <= port <= 65535 or not 1 <= timeout <= 180:
            raise ValueError("Invalid local model connection settings")
        self.model = model
        self.url = f"http://127.0.0.1:{port}/api/generate"
        self.timeout = timeout
        self.calls = 0
        self.seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.server_seconds = 0.0

    def __call__(self, task: str, text: str) -> str:
        if task not in TASKS:
            raise ValueError("Unsupported text task")
        prompt = (
            PROMPTS[task] + "\nReturn exactly one answer without explanations or quotes.\n"
            + f"Input: {text}\nOutput:"
        )
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0, "num_predict": 64}}).encode()
        request = Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        start = perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read(1_000_000))
        except (URLError, OSError, HTTPError) as exc:
            raise RuntimeError("Local Ollama unavailable. Start Ollama and download the requested model first.") from exc
        self.calls += 1
        self.seconds += perf_counter() - start
        for field, target in (("prompt_eval_count", "prompt_tokens"),
                              ("eval_count", "completion_tokens")):
            value = result.get(field)
            if type(value) is int and value >= 0:
                setattr(self, target, getattr(self, target) + value)
        duration = result.get("total_duration")
        if type(duration) is int and duration >= 0:
            self.server_seconds += duration / 1e9  # Ollama reports nanoseconds.
        if not isinstance(result, dict):
            raise ValueError("Local model response must be an object")
        answer = result.get("response")
        if not isinstance(answer, str):
            raise ValueError("Local model response lacked text")
        lines = answer.strip().strip('"').splitlines()
        if not lines or not lines[0].strip():
            raise ValueError("Local model response contained no usable answer")
        return lines[0].strip()


def run_ollama_demo(task: str, *, model: str, seed: int, output: str,
                    max_queries: int = 18, port: int = 11434) -> dict:
    """Learn from an actual local model; validate against independent rules."""
    if task not in TASKS:
        raise ValueError("Invalid task")
    values = generate(task, 120, seed)
    discovery, validation, test = values[:60], values[60:84], values[84:]
    teacher = OllamaTeacher(model, port=port)
    result = induce(task, lambda x: teacher(task, x), discovery, validation,
                    max_queries=max_queries, verifier=lambda x: oracle(task, x))
    heldout = (sum(result.skill.predict(x) == oracle(task, x) for x in test) / len(test)
               if result.skill else None)
    record = {
        "mode": "local_ollama_teacher", "model": model, "task": task,
        "status": result.status,
        "skill": result.skill.to_dict() if result.skill else None,
        "heldout_accuracy": heldout,
        "heldout_count": len(test),
        "model_calls": teacher.calls,
        "model_wall_seconds": round(teacher.seconds, 3),
        "validation": "against independent Python task oracle",
        "limitations": "This test uses small predefined transformations; it is not evidence of novel algorithm discovery.",
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "ollama_report.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if result.skill:
        (output_path / f"{task}.skill.json").write_text(
            json.dumps(result.skill.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return record
