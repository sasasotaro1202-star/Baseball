#!/usr/bin/env python3
"""
Unified Elo Baseline + Base Rate Comparison for Baseball and Soccer

実行方法:
  python elo_baseline_all.py

出力:
  - Elo ベースラインの精度
  - ベースレート (単純多数決) の精度
  - 改善幅 (Elo - Base Rate)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def calc_elo(df, k=24.0, home_adv=25.0, team_col='team', score_col='score', date_col='date'):
    """Elo レーティングを計算"""
    if df.empty:
        return df
    d = df.sort_values(date_col).reset_index(drop=True).copy()
    ratings = {}
    elo_team = []
    elo_opponent = []
    elo_expected = []
    
    for _, r in d.iterrows():
        team = r[team_col]
        opp = r.get('opponent', None)
        score = r[score_col]
        opp_score = r.get('opponent_score', None)
        
        team_elo = ratings.get(team, 1500.0)
        opp_elo = ratings.get(opp, 1500.0) if opp else 1500.0
        
        elo_team.append(team_elo)
        elo_opponent.append(opp_elo)
        
        expected = 1.0 / (1.0 + 10 ** (-((team_elo + home_adv) - opp_elo) / 400.0))
        elo_expected.append(expected)
        
        if score is not None and opp_score is not None:
            outcome = 1.0 if score > opp_score else (0.0 if score < opp_score else 0.5)
            ratings[team] = team_elo + k * (outcome - expected)
            if opp:
                ratings[opp] = opp_elo + k * ((1 - outcome) - (1 - expected))
    
    d['elo'] = elo_team
    d['elo_opponent'] = elo_opponent
    d['elo_expected'] = elo_expected
    return d


def evaluate_baseline(df, target_col, elo_expected_col='elo_expected'):
    """Elo ベースラインとベースレートを評価"""
    if df.empty or target_col not in df.columns:
        return None
    
    df = df.dropna(subset=[target_col]).copy()
    
    # Elo 予測
    if elo_expected_col in df.columns:
        df['pred_elo'] = (df[elo_expected_col] > 0.5).astype(int)
        elo_acc = (df['pred_elo'] == df[target_col]).mean()
    else:
        elo_acc = None
    
    # ベースレート (常に 1 と予測)
    base_rate = df[target_col].mean()
    
    # 常にホーム勝ちと予測
    if 'is_home' in df.columns:
        home_win_rate = (df[df['is_home'] == 1][target_col] == 1).mean() if len(df[df['is_home']==1]) > 0 else None
    else:
        home_win_rate = None
    
    return {
        'n_samples': len(df),
        'elo_accuracy': elo_acc,
        'base_rate': base_rate,
        'home_win_rate': home_win_rate,
        'elo_improvement': elo_acc - base_rate if elo_acc and pd.notna(elo_acc) else None
    }


def load_baseball_data():
    """MLB/NPB データを読み込む"""
    data_dir = Path('data/raw')
    games = []
    
    # MLB ゲーム
    for f in data_dir.glob('mlb_games_*.parquet'):
        try:
            df = pd.read_parquet(f)
            df['sport'] = 'mlb'
            games.append(df)
        except Exception:
            pass
    
    for f in data_dir.glob('mlb_games_*.csv'):
        try:
            df = pd.read_csv(f)
            df['sport'] = 'mlb'
            games.append(df)
        except Exception:
            pass
    
    # NPB ゲーム
    npb_file = data_dir / 'npb_games.parquet'
    if npb_file.exists():
        try:
            df = pd.read_parquet(npb_file)
            df['sport'] = 'npb'
            games.append(df)
        except Exception:
            pass
    
    return pd.concat(games, ignore_index=True) if games else pd.DataFrame()


def load_soccer_data():
    """サッカーデータを読み込む"""
    data_dir = Path('data/raw')
    games = []
    
    # football-data.co.uk
    fd_file = data_dir / 'football_data.parquet'
    if fd_file.exists():
        try:
            df = pd.read_parquet(fd_file)
            df['sport'] = 'soccer_fd'
            games.append(df)
        except Exception:
            pass
    
    for f in data_dir.glob('football_data_*.csv'):
        try:
            df = pd.read_csv(f)
            df['sport'] = 'soccer_fd'
            games.append(df)
        except Exception:
            pass
    
    # Understat
    us_file = data_dir / 'understat_matches.parquet'
    if us_file.exists():
        try:
            df = pd.read_parquet(us_file)
            df['sport'] = 'soccer_us'
            games.append(df)
        except Exception:
            pass
    
    return pd.concat(games, ignore_index=True) if games else pd.DataFrame()


def prepare_baseball_features(df):
    """野球データの特徴量を整備"""
    if df.empty:
        return df
    
    d = df.copy()
    
    # 日付パース
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
    
    # ホーム/アウェイを分離
    home_rows = []
    away_rows = []
    
    for _, r in d.iterrows():
        if pd.notna(r.get('home_score')) and pd.notna(r.get('away_score')):
            # ホーム視点
            home_rows.append({
                'date': r.get('date'),
                'team': r.get('home_team'),
                'opponent': r.get('away_team'),
                'score': r.get('home_score'),
                'opponent_score': r.get('away_score'),
                'is_home': 1,
                'target': int(r.get('home_score', 0) > r.get('away_score', 0)),
                'sport': r.get('sport', 'mlb'),
                'total_runs': r.get('home_score', 0) + r.get('away_score', 0)
            })
            # アウェイ視点
            away_rows.append({
                'date': r.get('date'),
                'team': r.get('away_team'),
                'opponent': r.get('home_team'),
                'score': r.get('away_score'),
                'opponent_score': r.get('home_score'),
                'is_home': 0,
                'target': int(r.get('away_score', 0) > r.get('home_score', 0)),
                'sport': r.get('sport', 'mlb'),
                'total_runs': r.get('home_score', 0) + r.get('away_score', 0)
            })
    
    return pd.DataFrame(home_rows + away_rows)


def prepare_soccer_features(df):
    """サッカーデータの特徴量を整備"""
    if df.empty:
        return df
    
    d = df.copy()
    
    # カラム名の正規化
    rename_map = {
        'FTHG': 'home_goals', 'FTAG': 'away_goals', 'FTR': 'result',
        'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'Date': 'date'
    }
    d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})
    
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], errors='coerce', dayfirst=True)
    
    home_rows = []
    away_rows = []
    
    for _, r in d.iterrows():
        if pd.notna(r.get('home_goals')) and pd.notna(r.get('away_goals')):
            # ホーム視点
            home_rows.append({
                'date': r.get('date'),
                'team': r.get('home_team'),
                'opponent': r.get('away_team'),
                'score': r.get('home_goals'),
                'opponent_score': r.get('away_goals'),
                'is_home': 1,
                'target': int(r.get('home_goals', 0) > r.get('away_goals', 0)),
                'sport': r.get('sport', 'soccer'),
                'total_goals': r.get('home_goals', 0) + r.get('away_goals', 0)
            })
            # アウェイ視点
            away_rows.append({
                'date': r.get('date'),
                'team': r.get('away_team'),
                'opponent': r.get('home_team'),
                'score': r.get('away_goals'),
                'opponent_score': r.get('home_goals'),
                'is_home': 0,
                'target': int(r.get('away_goals', 0) > r.get('home_goals', 0)),
                'sport': r.get('sport', 'soccer'),
                'total_goals': r.get('home_goals', 0) + r.get('away_goals', 0)
            })
    
    return pd.DataFrame(home_rows + away_rows)


def main():
    print("="*60)
    print("Elo Baseline + Base Rate Comparison")
    print("="*60)
    
    # データ読み込み
    print("\nLoading data...")
    bb_raw = load_baseball_data()
    soccer_raw = load_soccer_data()
    
    print(f"  Baseball games: {len(bb_raw)}")
    print(f"  Soccer games: {len(soccer_raw)}")
    
    # 特徴量準備
    print("\nPreparing features...")
    bb = prepare_baseball_features(bb_raw)
    soccer = prepare_soccer_features(soccer_raw)
    
    print(f"  Baseball rows (home+away): {len(bb)}")
    print(f"  Soccer rows (home+away): {len(soccer)}")
    
    results = {}
    
    # 野球 Elo 計算
    if not bb.empty:
        print("\n--- Baseball ---")
        bb = calc_elo(bb, team_col='team', score_col='score', date_col='date')
        
        # 全体
        res = evaluate_baseline(bb, 'target', 'elo_expected')
        if res:
            results['baseball_all'] = res
            print(f"  Overall (n={res['n_samples']}):")
            print(f"    Elo accuracy: {res['elo_accuracy']*100:.1f}%" if res['elo_accuracy'] else "    Elo: N/A")
            print(f"    Base rate: {res['base_rate']*100:.1f}%")
            print(f"    Improvement: +{res['elo_improvement']*100:.1f}pt" if res['elo_improvement'] else "    Improvement: N/A")
        
        # ハイ/ロー (合計 7 点以上)
        bb['target_high'] = (bb['total_runs'] >= 7).astype(int)
        res_high = evaluate_baseline(bb, 'target_high', 'elo_expected')
        if res_high:
            results['baseball_highlow'] = res_high
            print(f"\n  High/Low (n={res_high['n_samples']}):")
            print(f"    Base rate (high): {res_high['base_rate']*100:.1f}%")
    
    # サッカー Elo 計算
    if not soccer.empty:
        print("\n--- Soccer ---")
        soccer = calc_elo(soccer, team_col='team', score_col='score', date_col='date')
        
        res = evaluate_baseline(soccer, 'target', 'elo_expected')
        if res:
            results['soccer_all'] = res
            print(f"  Overall (n={res['n_samples']}):")
            print(f"    Elo accuracy: {res['elo_accuracy']*100:.1f}%" if res['elo_accuracy'] else "    Elo: N/A")
            print(f"    Base rate: {res['base_rate']*100:.1f}%")
            print(f"    Improvement: +{res['elo_improvement']*100:.1f}pt" if res['elo_improvement'] else "    Improvement: N/A")
    
    # 結果保存
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    output_dir = Path('data/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'elo_baseline.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_dir / 'elo_baseline.json'}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
