"""Regression tests for the honest offline comparison audit."""
import json
import pytest

from kitewatt.molt.competitive import AuditConfig, run_competitive_audit


@pytest.mark.parametrize("task", ["slug", "email_domain", "url_segment"])
def test_audit_uses_heldout_splits_and_measured_cache(tmp_path, task):
    report = run_competitive_audit(AuditConfig(
        task=task, discovery=12, validation=8, test=8,
        repeat=3, rounds=2, expected_requests=1000,
        output=str(tmp_path / task),
    ))
    assert report["partitions"] == {"discovery": 12, "validation": 8, "test": 8}
    assert report["compiler"]["teacher_calls"] >= 6
    assert report["heldout_test"]["exact_cache"]["heldout_accuracy"] == 1.0
    assert report["heldout_test"]["handwritten_reference"]["heldout_accuracy"] == 1.0
    assert report["heldout_test"]["cache_diagnostics"]["cold_teacher_calls_per_round"] == [8, 8]
    assert report["validation_decision"]["required_relative_gain"] == 0.2
    assert report["type"].startswith("offline")
    saved = json.loads((tmp_path / task / "competitive_report.json").read_text())
    assert saved["config"]["task"] == task


def test_audit_rejects_bad_config(tmp_path):
    with pytest.raises(ValueError):
        run_competitive_audit(AuditConfig(task="unknown", output=str(tmp_path)))
    with pytest.raises(ValueError):
        run_competitive_audit(AuditConfig(test=0, output=str(tmp_path)))


def test_audit_does_not_promote_inaccurate_teacher(tmp_path):
    report = run_competitive_audit(
        AuditConfig(discovery=12, validation=8, test=8, repeat=3,
                    rounds=2, expected_requests=1000, output=str(tmp_path)),
        teacher=lambda value: "wrong",
    )
    assert report["validation_decision"]["deploy"] is False
    assert report["heldout_test"]["teacher"]["heldout_accuracy"] < 1.0
    assert report["heldout_test"]["handwritten_reference"]["heldout_accuracy"] == 1.0
