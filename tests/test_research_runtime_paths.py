from pathlib import Path


def test_frozen_holdout_writes_to_repository_results():
    import research.frozen_holdout_gate as gate

    repo_root = Path(__file__).resolve().parents[1]
    assert gate.RESULTS == repo_root / "results"


def test_matchday_change_ledger_is_package_importable():
    import research.matchday_change_ledger as ledger

    assert ledger.RESULTS == Path("results")
