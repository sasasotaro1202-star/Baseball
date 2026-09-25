import re
import textwrap
from pathlib import Path


def test_recovery_embedded_python_compiles():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "baseball_recovery.yml"
    text = workflow.read_text(encoding="utf-8")
    blocks = re.findall(r"python3?\s+-\s*<<['\"]PY['\"]\n(.*?)\n\s*PY", text, re.DOTALL)
    assert blocks, "recovery workflow must contain a testable Python heredoc"
    for block in blocks:
        compile(textwrap.dedent(block), str(workflow), "exec")



def test_recovery_dispatches_independent_lanes_without_elif_serialization():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "baseball_recovery.yml"
    text = workflow.read_text(encoding="utf-8")
    start = text.index("                  # Dispatch independent lanes")
    end = text.index("\n\n          except Exception as exc:", start)
    block = text[start:end]
    assert "dispatched = []" in block
    for lane in ("validation", "matchday", "baseball-source-acquisition", "production-recovery", "mac-oos", "autonomous-research", "game-type-ablation"):
        assert lane in block
    assert "elif " not in block



def test_recovery_targets_current_npb_workflow_name():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "baseball_recovery.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "Baseball NPB Source Acquisition" in text


def test_recovery_starts_mlb_competition_when_no_prior_run_exists():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "baseball_recovery.yml"
    text = workflow.read_text(encoding="utf-8")
    start = text.index("if: ((") if "if: ((" in text else text.index("if ((mlb_competition")
    assert "mlb_competition is None" in text[start:start + 800]
