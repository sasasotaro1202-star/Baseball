from monitoring.recovery_controller import production_update_is_nonsemantic

def test_research_state_updates_are_nonsemantic_for_production():
    assert production_update_is_nonsemantic([
        "research_ai_advice.json",
        "research_history.json",
        "research_plan.json",
        "research_state.json",
    ])

def test_mixed_research_state_and_code_change_fails_closed():
    assert not production_update_is_nonsemantic([
        "research_state.json",
        "baseball_backtest.py",
    ])

def test_matchday_prefix_remains_nonsemantic():
    assert production_update_is_nonsemantic(["results/matchday_intelligence_current.json"])
