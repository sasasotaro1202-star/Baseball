# Drift-aware / Uncertainty-aware Expert Fusion

## 目的

現在のBaseball予測は、複数モデルを時系列検証損失の逆数で固定加重し、最後に単一の温度校正をかける構成です。
この方式は「全期間で同じ専門家比率」を仮定するため、シーズン途中の戦力・環境・データ生成過程の変化に対して routing が遅れる可能性があります。

今回の研究層では次の順序を採用します。

Historical model pool
→ past-only expert probabilities
→ drift-aware soft routing
→ uncertainty-aware weight smoothing
→ probability mixture
→ online recalibration
→ final probability

本研究層は本番予測を直接変更しません。OOS / frozen holdout / PIT / integrity gate を通過した候補だけが昇格対象です。

## 1. Drift-aware routing

各専門家 j の過去 LogLoss から、短期損失 S_j と長期損失 L_j を指数減衰で計算します。

R_j = (1 - λ_d) L_j + λ_d S_j

λ_d は予測時点で観測可能な drift score d の関数です。drift が弱いと長期安定性を重視し、drift が強いと直近適応性を重視します。

w_j ∝ exp(-R_j / T_d)

重要なのは「best model を1つ選ぶ」のではなく、soft routing のまま重みを連続変化させることです。急な regime change でも専門家を完全に切らず、fallback 能力を残します。

### Drift signal

最優先は、予測時点までに利用可能な入力特徴の reference/current window 差です。
現時点で完全な履歴特徴スナップショットがない場合は、研究用の output-space drift proxy として、過去の専門家予測分布の短期平均と長期平均の差を使います。

Observed outcome は drift signal に使用しません。

## 2. Uncertainty-aware routing

専門家間の予測不一致を epistemic uncertainty の近似として扱います。

U_disagreement = mean_j(std_k(p_jk))

さらに ensemble predictive entropy を加え、予測の曖昧さを 0..1 に正規化します。

不確実性が高い場合は routing 重みを一様分布へ一部戻します。

w'_j = (1-u) w_j + u/K

これにより、drift が強いときに1専門家へ過剰集中することと、専門家同士が割れているゲームで過信することを同時に抑えます。

## 3. Weight inertia

連続する試合で routing が振動しないよう、前回重みを慣性として混ぜます。

w_final = ρ w_previous + (1-ρ) w_current

drift が強いほど ρ は小さくなり、状態変化へ素早く反応します。
ただし専門家を hard switch しません。

## 4. Recalibration

routing 後の確率に対して temperature calibration を適用します。

q_k ∝ p_k^(1/T)

T は予測対象の outcome を見る前に更新してはいけません。
したがって運用順序は必ず

1. Predict using current T
2. Outcome becomes available after the game
3. Add the resolved row to calibration history
4. Fit/update T for subsequent predictions

です。

更新は指数移動平均＋最大ステップ制限で急変を抑えます。
現在の校正値を上書きしてしまうのではなく、直前値から安全に追従します。

## 5. OOS replay

research/routing_oos_replay.py は walk-forward checkpoint の expert_* 確率を使って、各試合を時系列順に再生します。

現在試合の prediction を生成する時点では、利用可能なのはそれ以前に解決済みの試合だけです。

比較対象:

- existing production-style ensemble
- drift-aware soft routing
- drift + uncertainty routing
- drift + uncertainty + online recalibration

評価:

- Accuracy
- LogLoss
- Brier
- ECE
- high-confidence subset
- low-confidence subset
- routing weight stability
- temperature stability

## 6. 本番昇格条件

以下の全条件を満たすまで研究専用とします。

- chronological OOS で改善
- 複数の時期 / regime で一貫
- frozen holdout で改善
- PIT / available_at 監査 PASS
- baseline より calibration が悪化しない
- 高 uncertainty 領域で過信が増えない
- routing weight が extreme oscillation しない
- missing expert / missing context に対して fail-closed
- production artifact integrity PASS
- validation workflow PASS
- production workflow PASS

単一期間の最高スコアだけでは昇格しません。

## 7. 研究上の重要点

### Static inverse-loss weighting からの進化

現行:
w_j ∝ 1 / validation_loss_j

新候補:
w_j(t) ∝ softmax(-[long_loss_j + drift_sensitive_short_term_adjustment_j])

つまり「モデル性能」だけでなく「今の環境で、その専門家の得意領域が続いているか」を扱います。

### Uncertainty と drift は別信号

- drift: データ生成過程が変わったか
- uncertainty: 今のゲームで専門家が一致しているか

drift が強くても専門家一致なら、必ずしも保守的に一様化しません。
逆に drift が弱くても専門家不一致なら、過信抑制を優先します。

### Market / Matchday Intelligence との接続

将来は Matchday Intelligence の starter / lineup / injury / weather / rest-travel / bullpen / market snapshot を、通常特徴として単純結合するのではなく、routing context としても利用可能です。

ただし historical replay では prediction-time snapshot が存在した場合だけ使用します。現在値を過去へ流用しません。

## 8. External research used

- Dynamic TMoE: drift detection + temporal-memory routing
- Online Bayesian Stacking: sequential log-likelihood based weight adaptation
- Online certificate-driven calibration: predict-then-update online calibration under distribution shift
- Conformal Online Model Aggregation: adaptive aggregation under drift instead of brittle model selection
- Uncertainty-driven dual-expert calibration: uncertainty-controlled routing + lightweight calibration
- Recent MLB public systems: per-model probabilities, model disagreement, lineup/starter/weather/context enrichment, and calibration diagnostics

All external methods are treated as design references. Fixed numeric caps/weights are not copied into production without Baseball-specific OOS validation.
