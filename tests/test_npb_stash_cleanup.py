from pathlib import Path

def test_npbruntime_empty_stash_cleanup_cannot_fail_success_push():
    text = Path(".github/workflows/baseball-parallel-source-acquisition.yml").read_text()
    assert "if git stash list | grep -q 'npb-runtime-ephemeral-before-push'; then" in text
    assert "git stash drop" in text
    assert "&& git stash drop" not in text
    assert "npb-runtime-ephemeral-before-failure-push" in text
