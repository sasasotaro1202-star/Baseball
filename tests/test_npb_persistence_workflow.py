from pathlib import Path

def test_npbruntime_push_path_stashes_ephemeral_changes():
    text = Path(".github/workflows/baseball-parallel-source-acquisition.yml").read_text()
    assert "git stash push --include-untracked -m 'npb-runtime-ephemeral-before-push'" in text
    assert "git rebase origin/main && git push origin HEAD:main" in text
    assert "git stash drop" in text
