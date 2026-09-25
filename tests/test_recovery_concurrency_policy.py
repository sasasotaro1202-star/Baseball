from pathlib import Path


def test_recovery_concurrency_isolation() -> None:
    workflow = Path(".github/workflows/baseball_recovery.yml").read_text(encoding="utf-8")
    assert "baseball-autonomous-recovery-${{ github.event_name }}-" in workflow
    assert "github.event.workflow_run.conclusion || 'scheduled'" in workflow
    assert "cancel-in-progress: true" in workflow
