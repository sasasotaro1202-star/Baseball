import pandas as pd

from research.international_asian_games_collector import parse_overview_html


def test_parse_asian_games_schedule_and_results():
    html = b"""
    <html><body>
      <h4>Opening Round</h4>
      <table>
        <tr><th>Date and time</th><th>Venue</th><th>Home - Visitor</th></tr>
        <tr><td>9/25/2026 18:30</td><td>Okazaki</td><td>Japan - Korea</td></tr>
      </table>
      <h4>Completed Round</h4>
      <table>
        <tr><th>Date and time</th><th>Venue</th><th>Home - Visitor</th></tr>
        <tr><td>9/23/2026 12:00</td><td>Toyohashi</td><td>Japan 7 - 0 Philippines</td></tr>
      </table>
    </body></html>
    """
    df = parse_overview_html(
        html,
        2026,
        "https://www.japan-baseball.jp/en/team/amateur/2026/asiangames/overview.html",
        "2026-09-25T07:00:00+00:00",
    )
    assert len(df) == 2
    assert set(df["status"]) == {"FINAL", "SCHEDULED"}
    assert set(df["international_game_category"]) == {"asian_games"}
    final = df.loc[df["status"] == "FINAL"].iloc[0]
    assert int(final["home_score"]) == 7
    assert int(final["away_score"]) == 0
    assert bool(final["prediction_features_allowed"]) is False
    assert final["pit_prediction_status"] == "NOT_A_PREDICTION_FEATURE_SOURCE"


def test_parser_rejects_unparseable_matchup():
    html = b"""
    <html><body>
      <h4>Opening Round</h4>
      <table>
        <tr><th>Date and time</th><th>Venue</th><th>Home - Visitor</th></tr>
        <tr><td>9/25/2026 18:30</td><td>Okazaki</td><td>TBD</td></tr>
      </table>
    </body></html>
    """
    try:
        parse_overview_html(html, 2026, "https://example.invalid", "2026-09-25T07:00:00+00:00")
    except RuntimeError as exc:
        assert "No Asian Games schedule rows parsed" in str(exc)
    else:
        raise AssertionError("unparseable schedule must fail closed")
