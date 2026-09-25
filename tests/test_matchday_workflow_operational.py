from pathlib import Path


def test_matchday_concurrency_is_latest_wins() -> None:
    workflow = Path(".github/workflows/baseball_matchday.yml").read_text(encoding="utf-8")
    assert "group: baseball-matchday" in workflow
    assert "cancel-in-progress: true" in workflow
