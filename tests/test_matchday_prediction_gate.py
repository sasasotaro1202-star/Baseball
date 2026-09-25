from production_matchday_intelligence import prediction_eligibility


def test_matchday_prediction_requires_verified_starters():
    ok, reason = prediction_eligibility({"starter_state": "UNKNOWN"}, True)
    assert ok is False
    assert reason == "STARTERS_UNCONFIRMED"


def test_matchday_prediction_requires_model():
    ok, reason = prediction_eligibility({"starter_state": "VERIFIED"}, False)
    assert ok is False
    assert reason == "MODEL_UNAVAILABLE"


def test_matchday_prediction_ready_only_with_model_and_starters():
    ok, reason = prediction_eligibility({"starter_state": "VERIFIED"}, True)
    assert ok is True
    assert reason == "READY"
