import re
from pathlib import Path


def test_recovery_embedded_python_compiles():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "baseball_recovery.yml"
    text = workflow.read_text(encoding="utf-8")
    blocks = re.findall(r"python3?\s+-\s*<<['\"]PY['\"]\n(.*?)\n\s*PY", text, re.DOTALL)
    assert blocks, "recovery workflow must contain a testable Python heredoc"
    for block in blocks:
        compile(block, str(workflow), "exec")
