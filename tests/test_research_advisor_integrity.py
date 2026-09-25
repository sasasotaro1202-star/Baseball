from pathlib import Path


def test_research_advisor_source_has_real_line_breaks():
    path = Path("research/openai_research_advisor.py")
    source = path.read_text(encoding="utf-8")

    assert source.startswith("#!/usr/bin/env python3\n")
    assert source.count("\n") >= 20
    assert "from __future__ import annotations\n" in source
    assert 'if __name__ == "__main__":\n' in source
