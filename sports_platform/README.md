# Sports Forecast Platform

野球（MLB/NPB）＋サッカー（Football-Data/Understat/SofaScore）の統合予測研究基盤。

## 機能

- **データ取得**: 増分収集・並列取得・HTTP キャッシュ・checkpoint/resume
- **品質ゲート**: 欠損・重複・異常値の自動検出
- **特徴量生成**: 先読み防止・リーク防止・Elo/ローリング集計
- **モデル研究**: 12 系統のモデルを walk-forward/rolling/expanding で総当たり
- **予測生成**: 期待値・Kelly 基準・ランキング出力
- **精度検証**: 実績との突合・劣化検知・自動改善フラグ
- **データ総量**: 観測単位（試合・打席・投球・イベント・特徴量）でカウント

## 使い方

```bash
pip install -r sports_platform/requirements.txt
cd sports_platform
python run.py collect      # 取得＋品質ゲート
python run.py features     # 特徴量生成
python run.py research     # バックテスト＋学習
python run.py predict      # 予測生成
python run.py verify       # 精度検証
python run.py inventory    # データ総量ダッシュボード
python run.py all          # 1 サイクル通し
```

## GitHub Actions

- `sports-historical-backfill.yml`: 過去データの並列バックフィル
- `sports-daily-update.yml`: 日次収集・予測・検証
- `sports-model-research.yml`: モデル研究・学習
- `sports-verify-inventory.yml`: 精度検証・データ総量比較
