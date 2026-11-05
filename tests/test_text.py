import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from kitewatt.molt.gate import decide
from kitewatt.molt.lab import LabConfig, generate, oracle, run_text_lab
from kitewatt.molt.text import TextProgram, TextRouter, TextSkill, induce, supported
from kitewatt.molt.web import handler_factory


def split(task, seed=7):
    samples = generate(task, 110, seed)
    return samples[:60], samples[60:84], samples[84:]


@pytest.mark.parametrize("task", ["slug", "email_domain", "url_segment"])
def test_active_induction_uses_disjoint_data_and_predicts_heldout(task):
    train, validation, test = split(task)
    assert not set(train) & set(validation)
    assert not set(test) & (set(train) | set(validation))
    result = induce(task, lambda value: oracle(task, value), train, validation)
    assert result.status == "compiled"
    assert result.skill is not None
    assert result.teacher_calls >= 6
    assert result.programs_examined > 1000
    assert all(result.skill.predict(value) == oracle(task, value) for value in test)


def test_text_program_is_data_not_code():
    with pytest.raises(ValueError):
        TextProgram(("__import__('os').system('echo not-safe')",))
    with pytest.raises(ValueError):
        TextProgram(("lower",) * 20)
    with pytest.raises(ValueError):
        TextProgram.from_dict({"operations": "lower"})
    assert TextProgram(("strip", "lower"))("  Hi  ") == "hi"


def test_skill_tampering_fails_validation():
    train, validation, _ = split("slug")
    skill = induce("slug", lambda value: oracle("slug", value), train, validation).skill
    assert skill
    data = skill.to_dict()
    assert TextSkill.from_dict(json.loads(json.dumps(data))).predict("  NEW TASK ") == "new-task"
    data["program"] = {"operations": ["reverse"]}
    with pytest.raises(ValueError, match="failed"):
        TextSkill.from_dict(data)


def test_domain_gate_and_fallback():
    assert not supported("email_domain", "not an email")
    assert not supported("url_segment", "file:///tmp/secret")
    assert not supported("slug", "bad/?query")
    router = TextRouter(lambda task, value: "fallback")
    train, val, _ = split("slug")
    skill = induce("slug", lambda value: oracle("slug", value), train, val).skill
    assert skill
    router.install(skill)
    assert router.predict("slug", "  Deep  Learning ") == "deep-learning"
    assert router.predict("slug", "bad/?query") == "fallback"
    assert router.counts["compiled"] == 1 and router.counts["fallback"] == 1


def test_compiler_rejects_unsupported_teacher_and_leakage():
    train, val, _ = split("slug")
    bad = induce("slug", lambda value: "arbitrary output", train, val)
    assert bad.skill is None
    assert bad.status == "no_matching_program"
    with pytest.raises(ValueError, match="disjoint"):
        induce("slug", lambda x: x, train, train[:10])


def test_cost_gate_refuses_slower_or_inaccurate_compilations():
    assert decide(reference_ms=10, compiled_ms=1, compile_seconds=0.1,
                  measured_accuracy=1.0).deploy
    bad = decide(reference_ms=1, compiled_ms=2, compile_seconds=0,
                 measured_accuracy=1.0)
    assert not bad.deploy and bad.reason == "not_faster_enough"
    bad = decide(reference_ms=10, compiled_ms=1, compile_seconds=1,
                 measured_accuracy=0.99)
    assert not bad.deploy and bad.reason == "accuracy_regression"
    bad = decide(reference_ms=10, compiled_ms=1, compile_seconds=10_000,
                 measured_accuracy=1.0, expected_requests=100)
    assert not bad.deploy and bad.reason == "not_amortized"


def test_full_text_lab_outputs_honest_cost_comparisons(tmp_path):
    report = run_text_lab(LabConfig(output=str(tmp_path), repeat=1))
    assert set(report["tasks"]) == {"slug", "email_domain", "url_segment"}
    for task, result in report["tasks"].items():
        assert result["status"] == "compiled"
        assert result["test_inputs_used_during_compilation"] == 0
        assert result["heldout_test_accuracy"] == 1.0
        assert result["cost_gate"] is not None
        assert (tmp_path / f"{task}.skill.json").exists()
    assert json.loads((tmp_path / "report.json").read_text())["benchmark"].startswith("MOLT")


def test_local_dashboard_health_and_restricted_input(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(tmp_path))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(origin + "/api/status", timeout=3) as response:
            status = json.load(response)
            assert status["paid_apis"] is False
        with urlopen(origin + "/", timeout=3) as response:
            assert b"Intelligence should" in response.read()
        request = Request(origin + "/api/predict", method="POST",
                          data=json.dumps({"task": "slug", "value": "  Deep Learning "}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=3) as response:
            assert json.load(response) == {"route": "reference", "answer": "deep-learning", "operations": None}
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_router_independent_spot_check_disables_a_bad_skill():
    train, validation, _ = split("slug")
    skill = induce("slug", lambda x: oracle("slug", x), train, validation).skill
    assert skill is not None
    router = TextRouter(lambda task, x: "REFERENCE", verifier=lambda task, x: "REFERENCE",
                        verify_every=1)
    router.install(skill)
    assert router.predict("slug", "  New Skill  ") == "REFERENCE"
    assert "slug" not in router.skills
    assert router.counts["verification_failures"] == 1
    assert router.predict("slug", "  Next Skill ") == "REFERENCE"


def test_optional_ollama_adapter_uses_only_local_host(monkeypatch):
    import io
    from kitewatt.molt.ollama import OllamaTeacher

    called = []
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            return json.dumps({"response": '"example.org"'}).encode()

    def fake_open(request, timeout):
        called.append((request.full_url, timeout, json.loads(request.data)))
        return FakeResponse()
    monkeypatch.setattr("kitewatt.molt.ollama.urlopen", fake_open)
    teacher = OllamaTeacher("local-model:1b", port=11434)
    assert teacher("email_domain", "User@Example.ORG") == "example.org"
    assert teacher.calls == 1
    assert called[0][0] == "http://127.0.0.1:11434/api/generate"
    assert called[0][2]["stream"] is False
    with pytest.raises(ValueError):
        OllamaTeacher("invalid\nname")
