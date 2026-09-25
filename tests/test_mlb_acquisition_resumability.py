from pathlib import Path


def test_mlb_acquisition_uses_resumable_date_chunks():
    text = Path("mlb_incremental_acquire.py").read_text(encoding="utf-8")
    assert "CHUNK_DAYS" in text
    assert "mlb_schedule_chunks" in text
    assert "def acquire_chunked" in text
    assert "resume existing chunk" in text
    assert "atomic_csv" in text
    assert "refusing to discard it" in text
