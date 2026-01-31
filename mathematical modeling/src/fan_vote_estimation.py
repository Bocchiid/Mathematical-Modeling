"""
MCM 2026 Problem C - Task 1
Fan Vote Estimation Model

Goal: Estimate unknown fan votes per contestant/week using judge scores
and elimination results, under rank-based or percent-based rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linear_sum_assignment, linprog
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
TOTAL_FAN_VOTES = 1_000_000  # per week, arbitrary scaling for estimates
EPSILON = 1e-4               # separation margin for optimization constraints
SMOOTH_WEIGHT = 0.6          # weight for week-to-week fan share smoothness
RANK_SCORE_DECAY = 0.35      # exponential decay for rank-to-vote conversion
MAX_WEEKS = 11
TIE_TOL = 1e-6
ELIM_BIAS = 1.5              # soft bias for eliminated to have worse combined rank
SURVIVOR_BIAS = 0.2          # soft bias to keep survivors from worst combined ranks
PENALTY_WEIGHT = 5000.0      # soft penalty for elimination constraint violations
RANK_SWAP_SPREAD = 0.0       # swap to enforce elimination in rank seasons
RANK_BLEND_CLOSE = 0.0       # shrink rank-based votes toward judge shares (close weeks)
RANK_BLEND_SPREAD = 3.0      # judge-spread threshold for applying shrinkage
RANK_BOTTOM_TWO_MARGIN = 1.0 # allow bottom-two match when ranks are very close


# -----------------------------
# Utilities
# -----------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.replace("N/A", np.nan)

    score_cols = [c for c in df.columns if "judge" in c]
    for c in score_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = ["celebrity_age_during_season", "season", "placement"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def impute_single_na_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    If exactly one judge score is missing in a week, fill it with
    the mean of the available judge scores for that week/contestant.
    """
    df = df.copy()
    for week in range(1, MAX_WEEKS + 1):
        cols = [f"week{week}_judge{i}_score" for i in range(1, 5)]
        for idx, row in df.iterrows():
            vals = [row.get(c) for c in cols]
            missing = [i for i, v in enumerate(vals) if pd.isna(v)]
            present = [v for v in vals if pd.notna(v)]
            if len(missing) == 1 and len(present) >= 2:
                fill = float(np.mean(present))
                df.at[idx, cols[missing[0]]] = fill
    return df


def get_voting_method(season: int) -> str:
    # Rank for seasons 1-2 and 28-34; Percent for 3-27
    return "rank" if season <= 2 or season >= 28 else "percent"


def uses_judge_save(season: int) -> bool:
    # Assumption: judge save used starting season 28
    return season >= 28




def judge_total(row: pd.Series, week: int) -> float:
    cols = [f"week{week}_judge{i}_score" for i in range(1, 5)]
    scores = [row[c] for c in cols if c in row.index and pd.notna(row[c])]
    scores = [s for s in scores if s != 0]
    return float(np.sum(scores)) if scores else np.nan


def week_contestants(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    season_df = df[df["season"] == season].copy()
    cols = [f"week{week}_judge{i}_score" for i in range(1, 5)]

    active_idx = []
    for idx, row in season_df.iterrows():
        scores = [row[c] for c in cols if c in row.index]
        if any(pd.notna(s) and s != 0 for s in scores):
            active_idx.append(idx)

    return season_df.loc[active_idx].copy()


def parse_elimination_week(results: str) -> int | None:
    if not isinstance(results, str):
        return None
    if results.startswith("Eliminated Week"):
        try:
            return int(results.split("Eliminated Week")[1].strip())
        except ValueError:
            return None
    return None


def last_active_week(row: pd.Series) -> int | None:
    active_weeks = []
    for week in range(1, MAX_WEEKS + 1):
        total = judge_total(row, week)
        if pd.notna(total) and total > 0:
            active_weeks.append(week)
    return max(active_weeks) if active_weeks else None


def elimination_weeks(df: pd.DataFrame) -> Dict[Tuple[int, str], int | None]:
    elim = {}
    for _, row in df.iterrows():
        season = int(row["season"])
        name = row["celebrity_name"]
        elim_week = parse_elimination_week(row["results"])

        if elim_week is None:
            if isinstance(row["results"], str) and "Withdrew" in row["results"]:
                elim_week = last_active_week(row)
            else:
                elim_week = None

        elim[(season, name)] = elim_week

    return elim


# -----------------------------
# Rank-based estimation
# -----------------------------

def _prior_rank_map(prior_votes: Dict[str, float] | None, names: List[str]) -> Dict[str, float]:
    if not prior_votes:
        return {n: np.nan for n in names}
    series = pd.Series({n: prior_votes.get(n, np.nan) for n in names})
    ranks = series.rank(ascending=False, method="average")
    return {n: float(ranks.loc[n]) for n in names}


def estimate_rank_week(
    week_df: pd.DataFrame,
    elim_names: List[str],
    prior_votes: Dict[str, float] | None = None,
    weight_prior: float = 0.4,
) -> Dict[str, float]:
    n = len(week_df)
    if n == 0:
        return {}

    week_df = week_df.copy()
    week_df["judge_total"] = week_df.apply(lambda r: judge_total(r, week_df["week"].iloc[0]), axis=1)

    # Rank: 1 = best (highest judge total)
    week_df["judge_rank"] = week_df["judge_total"].rank(ascending=False, method="average")

    spread = float(np.nanmax(week_df["judge_total"]) - np.nanmin(week_df["judge_total"]))

    # Soft assignment of fan ranks (1 best) using optimization
    names = week_df["celebrity_name"].tolist()
    ranks = list(range(1, n + 1))
    prior_rank = _prior_rank_map(prior_votes, names)

    cost = np.zeros((n, n))
    for i, name in enumerate(names):
        judge_r = float(week_df.loc[week_df["celebrity_name"] == name, "judge_rank"].iloc[0])
        for j, r in enumerate(ranks):
            prior_r = prior_rank.get(name, np.nan)
            prior_term = 0.0 if np.isnan(prior_r) else abs(r - prior_r)
            combined_rank = judge_r + r
            elim_term = 0.0
            survivor_term = 0.0
            if name in elim_names:
                # encourage higher combined rank for eliminated
                elim_term = -ELIM_BIAS * (combined_rank / (2 * n))
            else:
                # mildly discourage high combined rank for survivors
                survivor_term = SURVIVOR_BIAS * (combined_rank / (2 * n))
            cost[i, j] = abs(r - judge_r) + weight_prior * prior_term + elim_term + survivor_term

    row_ind, col_ind = linear_sum_assignment(cost)
    fan_ranks = {names[i]: ranks[j] for i, j in zip(row_ind, col_ind)}

    # Optional swap to enforce elimination when judge spread is large
    if spread >= RANK_SWAP_SPREAD and elim_names:
        judge_rank_map = {name: float(week_df.loc[week_df["celebrity_name"] == name, "judge_rank"].iloc[0])
                          for name in week_df["celebrity_name"].values}
        combined = {name: judge_rank_map[name] + fan_ranks[name] for name in week_df["celebrity_name"].values}
        elim_set = {name for name in elim_names if name in combined}
        if elim_set:
            k = max(1, len(elim_set))
            bottom_k = sorted(combined, key=combined.get, reverse=True)[:k]
            for elim in list(elim_set):
                if elim not in bottom_k:
                    worst = bottom_k[0]
                    fan_ranks[elim], fan_ranks[worst] = fan_ranks[worst], fan_ranks[elim]
                    combined[elim] = judge_rank_map[elim] + fan_ranks[elim]
                    combined[worst] = judge_rank_map[worst] + fan_ranks[worst]
                    bottom_k = sorted(combined, key=combined.get, reverse=True)[:k]

    # Convert ranks to fan vote shares
    # Higher rank -> higher votes, use exponential mapping
    rank_score = {name: np.exp(-RANK_SCORE_DECAY * (fan_ranks[name] - 1)) for name in week_df["celebrity_name"].values}
    total_score = sum(rank_score.values())

    # Judge-based share for shrinkage
    judge_total_vec = week_df["judge_total"].to_numpy()
    judge_share = judge_total_vec / np.nansum(judge_total_vec)
    judge_share_map = dict(zip(week_df["celebrity_name"].values, judge_share))

    blend = RANK_BLEND_CLOSE if spread <= RANK_BLEND_SPREAD else 0.0

    estimates = {}
    for name, score in rank_score.items():
        share = score / total_score
        blended = (1 - blend) * share + blend * judge_share_map[name]
        estimates[name] = blended * TOTAL_FAN_VOTES

    return estimates


# -----------------------------
# Percent-based estimation
# -----------------------------

def estimate_percent_week(
    week_df: pd.DataFrame,
    elim_names: List[str],
    prior_votes: Dict[str, float] | None = None,
    smooth_weight: float = SMOOTH_WEIGHT,
) -> Dict[str, float]:
    n = len(week_df)
    if n == 0:
        return {}

    week_df = week_df.copy()
    week_df["judge_total"] = week_df.apply(lambda r: judge_total(r, week_df["week"].iloc[0]), axis=1)

    judge_totals = week_df["judge_total"].to_numpy()
    J = np.nansum(judge_totals)
    judge_percent = judge_totals / J

    # Baseline fan shares: proportional to judge percent
    baseline = judge_percent.copy()

    # Prior fan shares from previous week (for smoothing)
    if prior_votes:
        prior = np.array([prior_votes.get(nm, np.nan) for nm in week_df["celebrity_name"].values], dtype=float)
        if np.all(np.isnan(prior)):
            prior = None
        else:
            prior = np.nan_to_num(prior, nan=np.nanmean(prior))
            prior = prior / np.sum(prior)
    else:
        prior = None

    # Optimization variables: fan share p_i >= 0, sum p_i = 1
    x0 = np.clip(baseline, 1e-6, 1.0)
    x0 = x0 / np.sum(x0)

    elim_mask = week_df["celebrity_name"].isin(elim_names).to_numpy()

    def objective(p):
        loss = np.sum((p - baseline) ** 2)
        if prior is not None:
            loss += smooth_weight * np.sum((p - prior) ** 2)
        return loss

    constraints = [
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0}
    ]

    # Hard elimination constraints to respect show rules
    for i in range(n):
        if not elim_mask[i]:
            continue
        for j in range(n):
            if elim_mask[j]:
                continue
            constraints.append({
                "type": "ineq",
                "fun": lambda p, i=i, j=j: (judge_percent[j] + p[j]) - (judge_percent[i] + p[i]) - EPSILON
            })

    bounds = [(0.0, 1.0) for _ in range(n)]

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    if not res.success:
        # Fallback: use baseline if optimization fails
        p = baseline
        p = p / np.sum(p)
    else:
        p = res.x

    estimates = {}
    for name, share in zip(week_df["celebrity_name"].values, p):
        estimates[name] = float(share * TOTAL_FAN_VOTES)

    return estimates


# -----------------------------
# Main estimation pipeline
# -----------------------------

@dataclass
class WeekEstimate:
    season: int
    week: int
    method: str
    fan_votes: Dict[str, float]
    eliminated: List[str]


def estimate_all_weeks(df: pd.DataFrame) -> List[WeekEstimate]:
    elim_map = elimination_weeks(df)
    results = []

    for season in sorted(df["season"].dropna().unique()):
        season = int(season)
        method = get_voting_method(season)
        prior_votes: Dict[str, float] = {}

        for week in range(1, MAX_WEEKS + 1):
            week_df = week_contestants(df, season, week)
            if week_df.empty:
                continue

            week_df = week_df.copy()
            week_df["week"] = week

            # Identify eliminated contestants in this week
            elim_names = [
                name for (s, name), w in elim_map.items()
                if s == season and w == week
            ]

            if method == "rank":
                fan_votes = estimate_rank_week(week_df, elim_names, prior_votes=prior_votes)
            else:
                fan_votes = estimate_percent_week(week_df, elim_names, prior_votes=prior_votes)

            results.append(WeekEstimate(
                season=season,
                week=week,
                method=method,
                fan_votes=fan_votes,
                eliminated=elim_names,
            ))

            # Update prior votes for next week (only for continuing contestants)
            prior_votes = {name: votes for name, votes in fan_votes.items()}

    return results


def consistency_metrics(estimates: List[WeekEstimate], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for est in estimates:
        if not est.fan_votes:
            continue

        week_df = week_contestants(df, est.season, est.week)
        if week_df.empty:
            continue

        week_df = week_df.copy()
        week_df["judge_total"] = week_df.apply(lambda r: judge_total(r, est.week), axis=1)

        if est.method == "rank":
            judge_rank = week_df["judge_total"].rank(ascending=False, method="average")
            fan_rank = pd.Series(est.fan_votes).reindex(week_df["celebrity_name"]).rank(ascending=False, method="average")
            combined = judge_rank.values + fan_rank.values
            order = np.argsort(combined)[::-1]
            worst_idx = int(order[0])
            predicted_elim = week_df["celebrity_name"].iloc[worst_idx]
            margin = combined[order[0]] - combined[order[1]] if len(order) >= 2 else np.nan
            bottom_two = set(week_df["celebrity_name"].iloc[order[:2]]) if len(order) >= 2 else {predicted_elim}
        else:
            judge_percent = week_df["judge_total"].to_numpy()
            judge_percent = judge_percent / np.nansum(judge_percent)
            fan_percent = np.array([est.fan_votes[n] for n in week_df["celebrity_name"].values])
            fan_percent = fan_percent / np.sum(fan_percent)
            combined = judge_percent + fan_percent
            worst_val = np.min(combined)
            worst_idx = int(np.argmin(combined))
            margin = np.partition(combined, 1)[1] - worst_val if len(combined) >= 2 else np.nan
            predicted_elim = week_df["celebrity_name"].iloc[worst_idx]
        actual_elim = est.eliminated[0] if est.eliminated else None
        correct = None
        if est.eliminated:
            if est.method == "rank" and uses_judge_save(est.season):
                correct = any(e in bottom_two for e in est.eliminated)
            elif est.method == "rank" and margin is not None and margin <= RANK_BOTTOM_TWO_MARGIN:
                correct = any(e in bottom_two for e in est.eliminated)
            else:
                correct = predicted_elim in est.eliminated

        rows.append({
            "season": est.season,
            "week": est.week,
            "method": est.method,
            "predicted_elimination": predicted_elim,
            "actual_elimination": actual_elim,
            "correct": correct,
            "margin": margin,
        })

    return pd.DataFrame(rows)


def feasible_width_percent(week_df: pd.DataFrame, elim_names: List[str]) -> pd.DataFrame:
    """
    Compute feasible interval widths for fan shares under percent rule.
    Returns per-contestant min/max and relative width.
    """
    n = len(week_df)
    if n == 0 or not elim_names:
        return pd.DataFrame()

    week_df = week_df.copy()
    week_df["judge_total"] = week_df.apply(lambda r: judge_total(r, week_df["week"].iloc[0]), axis=1)
    judge_percent = week_df["judge_total"].to_numpy()
    judge_percent = judge_percent / np.nansum(judge_percent)

    elim_mask = week_df["celebrity_name"].isin(elim_names).to_numpy()

    A = []
    b = []
    # For each eliminated i and survivor j: (judge_j + p_j) - (judge_i + p_i) >= EPS
    for i in range(n):
        if not elim_mask[i]:
            continue
        for j in range(n):
            if elim_mask[j]:
                continue
            row = np.zeros(n)
            row[j] = -1  # move to A p <= b form: -p_j + p_i <= -(judge_j - judge_i - EPS)
            row[i] = 1
            A.append(row)
            b.append(-(judge_percent[j] - judge_percent[i] - EPSILON))

    # Sum p = 1 -> two inequalities
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0) for _ in range(n)]

    rows = []
    for k in range(n):
        c = np.zeros(n)
        c[k] = 1.0

        # Minimize p_k
        res_min = linprog(c, A_ub=np.array(A) if A else None, b_ub=np.array(b) if b else None,
                          A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        # Maximize p_k -> minimize -p_k
        res_max = linprog(-c, A_ub=np.array(A) if A else None, b_ub=np.array(b) if b else None,
                          A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if res_min.success and res_max.success:
            p_min = res_min.x[k]
            p_max = res_max.x[k]
            p_mean = (p_min + p_max) / 2
            rel_width = (p_max - p_min) / p_mean if p_mean > 0 else np.nan
        else:
            p_min = p_max = rel_width = np.nan

        rows.append({
            "season": int(week_df["season"].iloc[0]),
            "week": int(week_df["week"].iloc[0]),
            "celebrity_name": week_df["celebrity_name"].iloc[k],
            "p_min": p_min,
            "p_max": p_max,
            "rel_width": rel_width,
        })

    return pd.DataFrame(rows)


def save_results(estimates: List[WeekEstimate], metrics: pd.DataFrame, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save fan vote estimates
    rows = []
    for est in estimates:
        for name, votes in est.fan_votes.items():
            rows.append({
                "season": est.season,
                "week": est.week,
                "method": est.method,
                "celebrity_name": name,
                "fan_votes": votes,
                "eliminated": name in est.eliminated,
            })

    pd.DataFrame(rows).to_csv(result_dir / "fan_vote_estimates.csv", index=False)

    # Save metrics
    metrics.to_csv(result_dir / "consistency_metrics.csv", index=False)

    # Plot accuracy by season
    if not metrics.empty:
        valid = metrics.dropna(subset=["correct"]).copy()
        if not valid.empty:
            acc = valid.groupby("season")["correct"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(acc["season"].astype(int), acc["correct"] * 100, color="#4C78A8")
            ax.set_title("Elimination Consistency by Season", fontsize=16, weight="bold")
            ax.set_xlabel("Season", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(result_dir / "accuracy_by_season.png", dpi=200)
            plt.close(fig)

        # Plot margin distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(valid["margin"].dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#F58518", color="#F58518"),
                   medianprops=dict(color="white", linewidth=2))
        ax.set_title("Elimination Margin Distribution (Confidence)", fontsize=16, weight="bold")
        ax.set_xlabel("Margin", fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(result_dir / "margin_distribution.png", dpi=200)
        plt.close(fig)

    # Plot relative width distribution (percent method only)
    if not metrics.empty:
        width_path = result_dir / "feasible_widths.csv"
        if width_path.exists():
            widths = pd.read_csv(width_path)
            widths = widths.dropna(subset=["rel_width"])
            if not widths.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(widths["rel_width"], bins=30, color="#54A24B", alpha=0.8)
                ax.set_title("Feasible Width (Relative) Distribution", fontsize=16, weight="bold")
                ax.set_xlabel("Relative Width", fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                plt.savefig(result_dir / "relative_width_distribution.png", dpi=200)
                plt.close(fig)


def run(csv_path: str, result_dir: Path) -> None:
    df = load_data(csv_path)
    df = impute_single_na_scores(df)
    estimates = estimate_all_weeks(df)
    metrics = consistency_metrics(estimates, df)

    print("\nFan vote estimation complete.")
    print("Total week estimates:", len(estimates))

    if not metrics.empty:
        valid = metrics.dropna(subset=["correct"])
        if not valid.empty:
            accuracy = valid["correct"].mean()
            print(f"Elimination consistency (accuracy): {accuracy:.2%}")
        else:
            print("No eliminations found for consistency check.")

    # Feasible width estimates (percent method only)
    width_rows = []
    for est in estimates:
        if est.method != "percent" or not est.eliminated:
            continue
        week_df = week_contestants(df, est.season, est.week)
        if week_df.empty:
            continue
        week_df = week_df.copy()
        week_df["week"] = est.week
        width_df = feasible_width_percent(week_df, est.eliminated)
        if not width_df.empty:
            width_rows.append(width_df)

    if width_rows:
        widths = pd.concat(width_rows, ignore_index=True)
        widths.to_csv(result_dir / "feasible_widths.csv", index=False)

    save_results(estimates, metrics, result_dir)
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    CSV_PATH = BASE_DIR / "data" / "2026_MCM_Problem_C_Data.csv"
    RESULT_DIR = BASE_DIR / "result"
    run(str(CSV_PATH), RESULT_DIR)
