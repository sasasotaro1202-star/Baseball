from lxml import html as lxml_html

from production_matchday_intelligence import _extract_official_starters


def test_official_starter_parser_rejects_generic_players_navigation():
    doc = lxml_html.fromstring(
        """
        <html><body>
          <h3>9月25日の予告先発投手</h3>
          <img alt="東京ヤクルトスワローズ">
          <a href="/bis/players/">個人年度別成績</a>
          <a href="/bis/players/63365153.html">山野　太一</a>
          <img alt="中日ドラゴンズ">
          <a href="/bis/players/">個人年度別成績</a>
          <a href="/bis/players/63395333.html">金丸　夢斗</a>
        </body></html>
        """
    )
    got = _extract_official_starters(doc, "9月25日の予告先発投手")
    assert got["東京ヤクルトスワローズ"] == "山野 太一"
    assert got["中日ドラゴンズ"] == "金丸 夢斗"


def test_official_starter_parser_fails_closed_without_target_heading():
    doc = lxml_html.fromstring(
        """
        <html><body>
          <h3>9月24日の予告先発投手</h3>
          <img alt="広島東洋カープ">
          <a href="/bis/players/63395332.html">森下　暢仁</a>
        </body></html>
        """
    )
    assert _extract_official_starters(doc, "9月25日の予告先発投手") == {}
