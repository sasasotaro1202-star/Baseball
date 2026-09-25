from pathlib import Path


def test_mlb_acquisition_uses_resumable_date_chunks():
    text = Path("mlb_incremental_acquire.py").read_text(encoding="utf-8")
    assert "CHUNK_DAYS" in text
    assert "mlb_schedule_chunks" in text
    assert "def acquire_chunked" in text
    assert "resume existing chunk" in text
    assert "atomic_csv" in text
    assert "refusing to discard it" in text


def test_zero_byte_chunk_is_quarantined_and_rebuilt(tmp_path, monkeypatch):
    import mlb_incremental_acquire as m
    import pandas as pd
    from datetime import date

    monkeypatch.setattr(m, "CHUNK_DIR", tmp_path)

    start = date(2020, 4, 15)
    end = date(2020, 5, 29)
    path = m.chunk_path(start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    monkeypatch.setattr(
        m,
        "fetch_schedule",
        lambda _start, _end: pd.DataFrame(columns=[
            "league", "game_id", "datetime", "home", "away",
            "home_score", "away_score", "home_starter", "away_starter",
            "venue", "game_type", "series_description", "confirmed_starters",
        ]),
    )

    out, ranges = m.acquire_chunked(start, end)

    assert len(out) == 0
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.with_suffix(path.suffix + ".corrupt").exists()
    assert ranges[0]["rows"] == 0


def test_empty_data_chunk_is_rebuilt(monkeypatch, tmp_path):
    import mlb_incremental_acquire as m
    import pandas as pd
    from datetime import date

    monkeypatch.setattr(m, "CHUNK_DIR", tmp_path)

    start = date(2020, 4, 15)
    end = date(2020, 5, 29)
    path = m.chunk_path(start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        m,
        "fetch_schedule",
        lambda _start, _end: pd.DataFrame(columns=[
            "league", "game_id", "datetime", "home", "away",
            "home_score", "away_score", "home_starter", "away_starter",
            "venue", "game_type", "series_description", "confirmed_starters",
        ]),
    )

    out, _ = m.acquire_chunked(start, end)
    assert len(out) == 0
    assert path.exists()



def test_single_newline_empty_checkpoint_is_valid_no_games_range(tmp_path, monkeypatch):
    import mlb_incremental_acquire as m
    from datetime import date

    monkeypatch.setattr(m, "CHUNK_DIR", tmp_path)
    start = date(2021, 11, 21)
    end = date(2022, 1, 4)
    path = m.chunk_path(start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n")

    calls = []
    monkeypatch.setattr(m, "fetch_schedule", lambda *_args: calls.append(True) or m.empty_chunk_frame())
    out, ranges = m.acquire_chunked(start, end)

    assert calls == []
    assert len(out) == 0
    assert ranges[0]["rows"] == 0


def test_mlb_acquisition_writes_production_readiness_status_contract():
    text = Path("mlb_incremental_acquire.py").read_text(encoding="utf-8")
    assert '"complete": True' in text
    assert '"aggregate_games"' in text
    assert "mlb_collection_status.json" in text
