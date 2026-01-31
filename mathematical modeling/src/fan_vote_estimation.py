"""
MCM 2026 Problem C - Task 1 (Forward-Generation Model)

Core idea:
- Generate fan vote shares from performance features + dynamic popularity random walk.
- No elimination results used in generation; only used in evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
TOTAL_VOTES = 1_000_000
MAX_WEEKS = 11
RANDOM_SEED = 42

# Model parameters (non-fitted, normalized)
BETA = np.array([1.0, 0.5, 0.3, -0.2, 0.1, 0.05])  # J_z, Jdiff_z, dJ_z, std_z, age_z, age2_z
RHO = 0.85
KAPPA = 0.25
SIGMA_A = 0.25
SIGMA_D = 0.20  # partner effect
SIGMA_H = 0.15  # industry effect
SIGMA_C = 0.10  # country/region effect

# Monte Carlo
MC_SIMS = 300

# Evaluation thresholds
RANK_BOTTOM_TWO_MARGIN = 1.0


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
    df = df.copy()
    for week in range(1, MAX_WEEKS + 1):
        cols = [f"week{week}_judge{i}_score" for i in range(1, 5)]
        for idx, row in df.iterrows():
            vals = [row.get(c) for c in cols]
            missing = [i for i, v in enumerate(vals) if pd.isna(v)]
            present = [v for v in vals if pd.notna(v)]
            if len(missing) == 1 and len(present) >= 2:
                df.at[idx, cols[missing[0]]] = float(np.mean(present))
    return df


def get_voting_method(season: int) -> str:
    return "rank" if season <= 2 or season >= 28 else "percent"


def uses_judge_save(season: int) -> bool:
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


def elimination_weeks(df: pd.DataFrame) -> Dict[Tuple[int, str], int | None]:
    elim = {}
    for _, row in df.iterrows():
        season = int(row["season"])
        name = row["celebrity_name"]
        elim_week = parse_elimination_week(row["results"])
        elim[(season, name)] = elim_week
    return elim


# -----------------------------
# Feature construction
# -----------------------------

def build_week_features(week_df: pd.DataFrame, week: int) -> pd.DataFrame:
    df = week_df.copy()
    df["judge_total"] = df.apply(lambda r: judge_total(r, week), axis=1)
    df["judge_mean"] = df["judge_total"].mean()
    df["judge_diff"] = df["judge_total"] - df["judge_mean"]

    if week == 1:
        df["delta_j"] = 0.0
        df["std_j"] = 0.0
    else:
        deltas = []
        stds = []
        for _, row in df.iterrows():
            prev = judge_total(row, week - 1)
            deltas.append(df.loc[row.name, "judge_total"] - prev if pd.notna(prev) else 0.0)

            history = []
            for w in range(1, week + 1):
                jt = judge_total(row, w)
                if pd.notna(jt):
                    history.append(jt)
            stds.append(float(np.std(history)) if len(history) >= 2 else 0.0)

        df["delta_j"] = deltas
        df["std_j"] = stds

    df["age"] = df["celebrity_age_during_season"].astype(float)
    df["age2"] = df["age"] ** 2

    for col in ["judge_total", "judge_diff", "delta_j", "std_j", "age", "age2"]:
        mu = df[col].mean()
        sd = df[col].std() if df[col].std() > 0 else 1.0
        df[col + "_z"] = (df[col] - mu) / sd

    return df


# -----------------------------
# Forward-generation model
# -----------------------------

@dataclass
class SimResult:
    season: int
    week: int
    names: List[str]
    p: np.ndarray
    a: np.ndarray


def simulate_season(df: pd.DataFrame, season: int, rng: np.random.Generator) -> List[SimResult]:
    results: List[SimResult] = []

    season_df = df[df["season"] == season].copy()
    partners = season_df["ballroom_partner"].dropna().unique().tolist()
    industries = season_df["celebrity_industry"].dropna().unique().tolist()
    countries = season_df["celebrity_homecountry/region"].dropna().unique().tolist()

    partner_effect = {p: rng.normal(0, SIGMA_D) for p in partners}
    industry_effect = {i: rng.normal(0, SIGMA_H) for i in industries}
    country_effect = {c: rng.normal(0, SIGMA_C) for c in countries}

    prev_a: Dict[str, float] = {}

    for week in range(1, MAX_WEEKS + 1):
        week_df = week_contestants(df, season, week)
        if week_df.empty:
            continue

        week_df = build_week_features(week_df, week)
        names = week_df["celebrity_name"].tolist()

        if week == 1:
            a = []
            for _, row in week_df.iterrows():
                mu = 0.2 * row["age_z"]
                mu += industry_effect.get(row["celebrity_industry"], 0.0)
                mu += country_effect.get(row["celebrity_homecountry/region"], 0.0)
                a.append(rng.normal(mu, 0.5))
            a = np.array(a)
        else:
            a = []
            for _, row in week_df.iterrows():
                name = row["celebrity_name"]
                prev = prev_a.get(name, 0.0)
                perf = row["judge_diff_z"]
                a.append(RHO * prev + KAPPA * perf + rng.normal(0, SIGMA_A))
            a = np.array(a)

        X = week_df[["judge_total_z", "judge_diff_z", "delta_j_z", "std_j_z", "age_z", "age2_z"]].to_numpy()
        d_partner = week_df["ballroom_partner"].map(partner_effect).fillna(0.0).to_numpy()
        h_ind = week_df["celebrity_industry"].map(industry_effect).fillna(0.0).to_numpy()

        u = X @ BETA + a + d_partner + h_ind
        expu = np.exp(u - np.max(u))
        p = expu / np.sum(expu)

        results.append(SimResult(season=season, week=week, names=names, p=p, a=a))
        prev_a = {name: val for name, val in zip(names, a)}

    return results


# -----------------------------
# Monte Carlo aggregation
# -----------------------------

def run_simulations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    all_records = []
    all_a = []

    for b in range(MC_SIMS):
        for season in sorted(df["season"].dropna().unique()):
            season = int(season)
            sim_results = simulate_season(df, season, rng)
            for res in sim_results:
                for name, p_val, a_val in zip(res.names, res.p, res.a):
                    all_records.append({
                        "sim": b,
                        "season": res.season,
                        "week": res.week,
                        "celebrity_name": name,
                        "p": p_val,
                        "votes": p_val * TOTAL_VOTES,
                    })
                    all_a.append({
                        "sim": b,
                        "season": res.season,
                        "week": res.week,
                        "celebrity_name": name,
                        "a": a_val,
                    })

    sims = pd.DataFrame(all_records)
    a_df = pd.DataFrame(all_a)

    summary = sims.groupby(["season", "week", "celebrity_name"]).agg(
        p_mean=("p", "mean"),
        p_sd=("p", "std"),
        p_q025=("p", lambda x: np.quantile(x, 0.025)),
        p_q975=("p", lambda x: np.quantile(x, 0.975)),
    ).reset_index()

    summary["p_cv"] = summary["p_sd"] / summary["p_mean"].replace(0, np.nan)
    summary["votes_mean"] = summary["p_mean"] * TOTAL_VOTES

    a_summary = a_df.groupby(["season", "week", "celebrity_name"]).agg(
        a_mean=("a", "mean"),
        a_sd=("a", "std"),
    ).reset_index()

    return summary, sims, a_summary


# -----------------------------
# Evaluation (test stage)
# -----------------------------

def evaluate_consistency(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    elim_map = elimination_weeks(df)
    rows = []

    for season in sorted(df["season"].dropna().unique()):
        season = int(season)
        method = get_voting_method(season)

        for week in range(1, MAX_WEEKS + 1):
            week_df = week_contestants(df, season, week)
            if week_df.empty:
                continue

            week_df = build_week_features(week_df, week)
            names = week_df["celebrity_name"].tolist()
            p = summary[(summary["season"] == season) & (summary["week"] == week)]
            p = p.set_index("celebrity_name").reindex(names)["p_mean"].to_numpy()

            if method == "percent":
                judge_percent = week_df["judge_total"].to_numpy()
                judge_percent = judge_percent / np.nansum(judge_percent)
                combined = judge_percent + p
                order = np.argsort(combined)
                pred = names[order[0]]
                margin = combined[order[1]] - combined[order[0]] if len(order) >= 2 else np.nan
                bottom_two = {names[order[0]], names[order[1]]} if len(order) >= 2 else {pred}
            else:
                judge_rank = week_df["judge_total"].rank(ascending=False, method="average").to_numpy()
                fan_rank = pd.Series(p, index=names).rank(ascending=False, method="average").to_numpy()
                combined = judge_rank + fan_rank
                order = np.argsort(combined)[::-1]
                pred = names[order[0]]
                margin = combined[order[0]] - combined[order[1]] if len(order) >= 2 else np.nan
                bottom_two = {names[order[0]], names[order[1]]} if len(order) >= 2 else {pred}

            actual = [name for (s, name), w in elim_map.items() if s == season and w == week]
            if not actual:
                correct = None
            else:
                if method == "rank" and (uses_judge_save(season) or (margin is not None and margin <= RANK_BOTTOM_TWO_MARGIN)):
                    correct = any(a in bottom_two for a in actual)
                else:
                    correct = pred in actual

            rows.append({
                "season": season,
                "week": week,
                "method": method,
                "predicted_elimination": pred,
                "actual_elimination": actual[0] if actual else None,
                "correct": correct,
                "margin": margin,
            })

    return pd.DataFrame(rows)


def elimination_probability(df: pd.DataFrame, sims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(df["season"].dropna().unique()):
        season = int(season)
        method = get_voting_method(season)

        for week in range(1, MAX_WEEKS + 1):
            week_df = week_contestants(df, season, week)
            if week_df.empty:
                continue

            names = week_df["celebrity_name"].tolist()
            p = sims[(sims["season"] == season) & (sims["week"] == week)]
            if p.empty:
                continue

            judge_tot = week_df.apply(lambda r: judge_total(r, week), axis=1).to_numpy()
            judge_percent = judge_tot / np.nansum(judge_tot)
            judge_rank = pd.Series(judge_tot).rank(ascending=False, method="average").to_numpy()

            for sim_id, group in p.groupby("sim"):
                p_vec = group.set_index("celebrity_name").reindex(names)["p"].to_numpy()
                if method == "percent":
                    combined = judge_percent + p_vec
                    pred = names[int(np.argmin(combined))]
                else:
                    fan_rank = pd.Series(p_vec, index=names).rank(ascending=False, method="average").to_numpy()
                    combined = judge_rank + fan_rank
                    pred = names[int(np.argmax(combined))]
                rows.append({"season": season, "week": week, "sim": sim_id, "pred": pred})

    prob = pd.DataFrame(rows).groupby(["season", "week", "pred"]).size().reset_index(name="count")
    prob["prob"] = prob["count"] / MC_SIMS
    return prob


# -----------------------------
# Plotting
# -----------------------------

def save_plots(df: pd.DataFrame, summary: pd.DataFrame, a_summary: pd.DataFrame,
               metrics: pd.DataFrame, elim_prob: pd.DataFrame, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    # A1: contestants per season
    season_counts = df.groupby("season")["celebrity_name"].nunique().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(season_counts["season"], season_counts["celebrity_name"], color="#9ECBF3")
    ax.set_title("Contestants per Season", fontsize=14, weight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(result_dir / "contestants_per_season.png", dpi=200)
    plt.close(fig)

    # A2: weeks per season
    weeks = []
    for season in sorted(df["season"].dropna().unique()):
        season = int(season)
        w = 0
        for week in range(1, MAX_WEEKS + 1):
            if not week_contestants(df, season, week).empty:
                w += 1
        weeks.append({"season": season, "weeks": w})
    weeks_df = pd.DataFrame(weeks)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weeks_df["season"], weeks_df["weeks"], marker="o", color="#F2B57A")
    ax.set_title("Weeks per Season", fontsize=14, weight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Weeks")
    plt.tight_layout()
    plt.savefig(result_dir / "weeks_per_season.png", dpi=200)
    plt.close(fig)

    # A3: judge total distribution
    judge_totals = []
    for season in sorted(df["season"].dropna().unique()):
        season = int(season)
        for week in range(1, MAX_WEEKS + 1):
            week_df = week_contestants(df, season, week)
            if week_df.empty:
                continue
            week_df = build_week_features(week_df, week)
            judge_totals.extend(week_df["judge_total"].tolist())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(judge_totals, bins=30, color="#A7D9A6", alpha=0.85)
    ax.set_title("Judge Total Score Distribution", fontsize=14, weight="bold")
    ax.set_xlabel("Total Score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(result_dir / "judge_total_distribution.png", dpi=200)
    plt.close(fig)

    # A4: judge totals by season (boxplot)
    by_season = []
    for season in sorted(df["season"].dropna().unique()):
        season = int(season)
        vals = []
        for week in range(1, MAX_WEEKS + 1):
            week_df = week_contestants(df, season, week)
            if week_df.empty:
                continue
            week_df = build_week_features(week_df, week)
            vals.extend(week_df["judge_total"].tolist())
        by_season.append(vals)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(by_season, patch_artist=True, boxprops=dict(facecolor="#AFCDF7", color="#AFCDF7"))
    ax.set_title("Judge Scores by Season", fontsize=14, weight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Total Score")
    plt.tight_layout()
    plt.savefig(result_dir / "judge_scores_by_season.png", dpi=200)
    plt.close(fig)

    # B1: heatmap of p for selected season
    sel_season = int(df["season"].max())
    sub = summary[summary["season"] == sel_season].copy()
    pivot = sub.pivot(index="celebrity_name", columns="week", values="p_mean").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_title(f"Vote Share Heatmap (Season {sel_season})", fontsize=14, weight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    plt.savefig(result_dir / "vote_share_heatmap.png", dpi=200)
    plt.close(fig)

    # B2: top-3 vote share time series (selected season)
    season_df = df[df["season"] == sel_season]
    top3 = season_df.sort_values("placement").head(3)["celebrity_name"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in top3:
        series = sub[sub["celebrity_name"] == name].sort_values("week")
        ax.plot(series["week"], series["p_mean"], marker="o", label=name)
        ax.fill_between(series["week"], series["p_q025"], series["p_q975"], alpha=0.2)
    ax.set_title(f"Top-3 Vote Share with 95% CI (Season {sel_season})", fontsize=14, weight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Vote Share")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(result_dir / "top3_vote_share_ci.png", dpi=200)
    plt.close(fig)

    # B3: popularity trajectories (champion, mid, early)
    placements = season_df.dropna(subset=["placement"]).sort_values("placement")
    champion = placements.iloc[0]["celebrity_name"]
    mid = placements.iloc[len(placements) // 2]["celebrity_name"]
    early = placements.iloc[-1]["celebrity_name"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in zip([champion, mid, early], ["#8FBCEB", "#F2B57A", "#F6A6A6"]):
        series = a_summary[(a_summary["season"] == sel_season) & (a_summary["celebrity_name"] == name)].sort_values("week")
        ax.plot(series["week"], series["a_mean"], marker="o", label=name, color=color)
    ax.set_title(f"Popularity Trajectories (Season {sel_season})", fontsize=14, weight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Popularity a")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(result_dir / "popularity_trajectories.png", dpi=200)
    plt.close(fig)

    # C1: accuracy by season
    valid = metrics.dropna(subset=["correct"]).copy()
    if not valid.empty:
        acc = valid.groupby("season")["correct"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(acc["season"], acc["correct"] * 100, color="#B7E0DC")
        ax.set_title("Elimination Consistency by Season", fontsize=14, weight="bold")
        ax.set_xlabel("Season")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(result_dir / "accuracy_by_season.png", dpi=200)
        plt.close(fig)

    # C2: margin distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(valid["margin"].dropna(), vert=False, patch_artist=True,
               boxprops=dict(facecolor="#F7C08B", color="#F7C08B"),
               medianprops=dict(color="white", linewidth=2))
    ax.set_title("Elimination Margin Distribution", fontsize=14, weight="bold")
    ax.set_xlabel("Margin")
    plt.tight_layout()
    plt.savefig(result_dir / "margin_distribution.png", dpi=200)
    plt.close(fig)

    # D1: width heatmap
    width = summary.copy()
    width["width"] = width["p_q975"] - width["p_q025"]
    pivot_w = width[width["season"] == sel_season].pivot(index="celebrity_name", columns="week", values="width").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot_w.values, aspect="auto", cmap="YlOrBr")
    ax.set_title(f"Width Heatmap (Season {sel_season})", fontsize=14, weight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant")
    ax.set_yticks(range(len(pivot_w.index)))
    ax.set_yticklabels(pivot_w.index, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    plt.savefig(result_dir / "width_heatmap.png", dpi=200)
    plt.close(fig)

    # D2: width vs margin scatter
    width_week = summary.copy()
    width_week["width"] = width_week["p_q975"] - width_week["p_q025"]
    width_week = width_week.groupby(["season", "week"])["width"].mean().reset_index()
    merged = metrics.merge(width_week, on=["season", "week"], how="left")
    if not merged.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(merged["margin"], merged["width"], s=10, alpha=0.5, color="#9ECBF3")
        ax.set_title("Width vs Margin", fontsize=14, weight="bold")
        ax.set_xlabel("Margin")
        ax.set_ylabel("Width")
        plt.tight_layout()
        plt.savefig(result_dir / "width_vs_margin.png", dpi=200)
        plt.close(fig)

    # D3: elimination probability over weeks (selected season)
    prob = elim_prob[elim_prob["season"] == sel_season]
    fig, ax = plt.subplots(figsize=(10, 5))
    for week in sorted(prob["week"].unique()):
        subp = prob[prob["week"] == week].sort_values("prob", ascending=False).head(1)
        ax.bar(week, subp["prob"].iloc[0], color="#F3A6A6")
    ax.set_title(f"Top Elimination Probability by Week (Season {sel_season})", fontsize=14, weight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Probability")
    plt.tight_layout()
    plt.savefig(result_dir / "elim_prob_by_week.png", dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def run(csv_path: str, result_dir: Path) -> None:
    df = load_data(csv_path)
    df = impute_single_na_scores(df)

    summary, sims, a_summary = run_simulations(df)
    metrics = evaluate_consistency(df, summary)
    elim_prob = elimination_probability(df, sims)

    result_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(result_dir / "fan_vote_estimates.csv", index=False)
    metrics.to_csv(result_dir / "consistency_metrics.csv", index=False)
    a_summary.to_csv(result_dir / "popularity_estimates.csv", index=False)
    elim_prob.to_csv(result_dir / "elimination_probability.csv", index=False)

    save_plots(df, summary, a_summary, metrics, elim_prob, result_dir)

    valid = metrics.dropna(subset=["correct"])
    if not valid.empty:
        acc = valid["correct"].mean()
        print(f"Elimination consistency (accuracy): {acc:.2%}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    CSV_PATH = BASE_DIR / "data" / "2026_MCM_Problem_C_Data.csv"
    RESULT_DIR = BASE_DIR / "result"
    run(str(CSV_PATH), RESULT_DIR)
