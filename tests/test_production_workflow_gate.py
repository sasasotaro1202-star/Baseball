from pathlib import Path


def test_production_workflow_validation_gate_is_race_safe():
    text = Path(".github/workflows/baseball_production.yml").read_text(encoding="utf-8")
    expected = "if: github.event_name != 'workflow_run' && steps.sha_gate.outputs.ok == 'true'"
    assert expected in text, "manual/scheduled validation check must remain gated"
    assert "FAIL-CLOSED: upstream validation conclusion" in text, "workflow_run validation must fail closed when upstream validation is not successful"
    assert "Verify validation SHA matches current main" in text
    assert "Require successful validation for current main" in text


def test_production_downstream_steps_are_guarded_by_sha_gate():
    text = Path(".github/workflows/baseball_production.yml").read_text(encoding="utf-8")
    for marker in (
        "Hosted runner health gate",
        "Setup Python",
        "Preflight and integrity gate",
        "NPB readiness gate and chronological OOS",
        "MLB data acquisition, starter quality gate and chronological OOS",
        "Validate OOS predictions before any state write",
        "Persist verified state atomically",
    ):
        pos = text.index(marker)
        block = text[pos:pos + 1800]
        assert "steps.sha_gate.outputs.ok == 'true'" in block, f"{marker} is not guarded by the SHA gate"


def test_production_readiness_is_independent_of_npb_status():
    text = Path(".github/workflows/baseball_production.yml").read_text(encoding="utf-8")
    assert "data/checkpoints/mlb_collection_status.json" in text
    assert "data/checkpoints/npb_collection_status.json" not in text
    assert "steps.mlb_ready.outputs.ready == 'true'" in text
