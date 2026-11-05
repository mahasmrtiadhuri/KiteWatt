"""Guard against artificial savings, leakage, and broken local-model integration."""
import json
from pathlib import Path

import pytest

from kitewatt.molt.lab import generate, oracle
from kitewatt.molt.workloads import WorkloadConfig, load_dataset, run_workload


def test_offline_workload_has_independent_test_and_cache_baseline(tmp_path):
    report = run_workload(WorkloadConfig(output=str(tmp_path), repeat=3))
    assert report["partitions"] == {"discovery": 48, "validation": 20, "test": 24}
    assert report["comparison"]["heldout_split_unseen_during_induction_and_gate"]
    assert report["compiler"]["status"] == "compiled"
    assert report["model_reference"]["heldout_accuracy"] == 1.0
    assert report["compiled_skill"]["heldout_accuracy"] == 1.0
    assert report["exact_cache"]["cold_model_calls"] == 24
    assert report["exact_cache"]["warm_cache_hits"] == 48
    assert report["model_reference"]["server_prompt_tokens_observed"] is None
    assert report["model_reference"]["teacher_calls_during_gate_calibration"] == 20
    assert json.loads((tmp_path / "workload_report.json").read_text())["config"]["backend"] == "reference"
    assert (tmp_path / "skill.json").exists()


def test_model_errors_do_not_get_hidden_by_compilation(tmp_path):
    seen = []
    def unhelpful_teacher(text):
        seen.append(text)
        return "not-a-valid-rule"
    report = run_workload(WorkloadConfig(output=str(tmp_path)), teacher=unhelpful_teacher)
    assert report["compiler"]["status"] == "no_matching_program"
    assert report["compiled_skill"]["deployed"] is False
    assert report["compiled_skill"]["heldout_accuracy"] is None
    assert report["model_reference"]["heldout_accuracy"] == 0
    assert len(seen) == report["model_reference"]["total_teacher_calls_observed"]


def test_dataset_has_explicit_disjoint_splits_and_no_model_calls_for_validation(tmp_path):
    path = tmp_path / "workloads.jsonl"
    samples = generate("email_domain", 40, 31)
    rows = []
    for i, value in enumerate(samples):
        rows.append({"task": "email_domain", "input": value,
                     "split": "discovery" if i < 16 else "validation" if i < 28 else "test"})
    path.write_text("\n".join(json.dumps(row) for row in rows))
    discovery, valid, test = load_dataset(path, "email_domain")
    assert len(discovery) == 16 and len(valid) == 12 and len(test) == 12
    seen = []
    def teacher(value):
        seen.append(value)
        return oracle("email_domain", value)
    report = run_workload(WorkloadConfig(task="email_domain", output=str(tmp_path / "result"),
                                         dataset_file=str(path), max_queries=8), teacher=teacher)
    assert report["compiler"]["status"] == "compiled"
    assert not (set(seen[:report["compiler"]["teacher_queries"]]) & (set(test) | set(valid)))
    assert report["partitions"]["test"] == 12
    path.write_text(path.read_text() + "\n" + json.dumps(rows[0]))
    with pytest.raises(ValueError, match="disjoint"):
        load_dataset(path, "email_domain")


def test_dataset_and_config_validation(tmp_path):
    with pytest.raises(ValueError, match="configuration"):
        run_workload(WorkloadConfig(test=0))
    p = tmp_path / "test.jsonl"
    p.write_text('{"task":"slug","split":"test","input":"Hi World", "leak":true}')
    with pytest.raises(ValueError, match="schema"):
        load_dataset(p, "slug")
    with pytest.raises(ValueError, match="Dataset"):
        load_dataset(tmp_path / "missing", "slug")


def test_ollama_backend_uses_local_only_and_tracks_tokens(monkeypatch, tmp_path):
    from kitewatt.molt.ollama import OllamaTeacher
    import kitewatt.molt.ollama as module

    def fake_open(request, timeout):
        body = json.loads(request.data)
        assert request.full_url == "http://127.0.0.1:11434/api/generate"
        value = body["prompt"].split("Input: ", 1)[1].split("\nOutput:", 1)[0]
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size):
                return json.dumps({"response": oracle("slug", value),
                                   "prompt_eval_count": 42, "eval_count": 6,
                                   "total_duration": 50_000_000}).encode()
        return Response()
    monkeypatch.setattr(module, "urlopen", fake_open)
    report = run_workload(WorkloadConfig(backend="ollama", output=str(tmp_path)))
    stats = report["model_reference"]
    assert stats["heldout_accuracy"] == 1.0
    assert stats["total_teacher_calls_observed"] >= 24
    assert stats["server_prompt_tokens_observed"] == stats["total_teacher_calls_observed"] * 42
    assert stats["server_generated_tokens_observed"] == stats["total_teacher_calls_observed"] * 6
    assert stats["server_total_duration_seconds_observed"] == pytest.approx(stats["total_teacher_calls_observed"] * 0.05)
    assert report["compiled_skill"]["heldout_accuracy"] == 1.0


def test_local_dashboard_workload_endpoint(tmp_path):
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request, urlopen
    from kitewatt.molt.web import handler_factory
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(tmp_path))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(origin + "/api/status", timeout=3) as response:
            payload = json.load(response)
            assert payload["version"] == "2.0.0"
            assert "model_bench" in payload["modes"]
        with urlopen(origin, timeout=3) as response:
            assert b"Local-model benchmark" in response.read()
        request = Request(origin + "/api/model-bench", method="POST",
                          data=json.dumps({"task": "slug", "backend": "reference", "test_count": 8}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=20) as response:
            report = json.load(response)
            assert report["partitions"]["test"] == 8
            assert report["model_reference"]["heldout_accuracy"] == 1
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
