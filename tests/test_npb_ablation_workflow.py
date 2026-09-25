from pathlib import Path


def test_npb_ablation_uses_seasonal_sources_and_compare_dependencies():
    text = Path(".github/workflows/baseball_game_type_ablation.yml").read_text()
    assert 'glob("*_multi_source_pbp.csv")' in text
    assert "Setup Python for comparison" in text
    assert "pip install --disable-pip-version-check -r requirements.txt" in text
    assert "python research/compare_game_type_ablation.py --root ablation_artifacts" in text
