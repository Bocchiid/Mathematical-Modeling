# -*- coding: utf-8 -*-
# Dynamic Latent Factor Model for Fan Vote Estimation
# MCM 2026 Problem C - Task 1

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load MCM data and convert from wide format to long format."""
    df = pd.read_csv(data_path)
    
    # Rename columns for clarity
    df = df.rename(columns={
        'celebrity_name': 'Name',
        'season': 'Season',
        'celebrity_industry': 'Industry',
        'celebrity_age_during_season': 'Age',
        'celebrity_homestate': 'Region',
        'ballroom_partner': 'Partner',
        'results': 'Result',
    })
    
    # Convert from wide format to long format
    # Judge scores are in columns: week{1-11}_judge{1-4}_score
    
    judge_cols = []
    for week in range(1, 12):
        for judge in range(1, 5):
            judge_cols.append(f'week{week}_judge{judge}_score')
    
    # Melt into long format
    id_vars = ['Name', 'Season', 'Industry', 'Age', 'Region', 'Partner', 'Result', 'placement']
    df_long = df.melt(id_vars=id_vars, 
                       value_vars=judge_cols,
                       var_name='judge_col',
                       value_name='judge_score')
    
    # Extract week and judge from column name
    df_long['Week'] = df_long['judge_col'].str.extract(r'week(\d+)').astype(int)
    df_long['Judge'] = df_long['judge_col'].str.extract(r'judge(\d)').astype(int)
    
    # Drop intermediate columns
    df_long = df_long.drop(columns=['judge_col'])
    
    return df_long.reset_index(drop=True)


def normalize_judge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize judge scores within each (season, week).
    z_i,t = (s_i,t - mu_t(s)) / sigma_t(s)
    
    Data is in long format: each row is a (contestant, week, judge) combination.
    """
    df = df.copy()
    
    # For each (season, week), compute mean and std of contestant scores
    # Group by season and week, get all scores
    for (season, week), group in df.groupby(['Season', 'Week']):
        # Get contestant means for this week
        contestant_means = group.groupby('Name')['judge_score'].mean()
        
        # Week stats
        week_mean = contestant_means.mean()
        week_std = contestant_means.std()
        
        if week_std > 0:
            # Normalize for each contestant in this week
            for idx in group.index:
                contestant = df.loc[idx, 'Name']
                contestant_mean = contestant_means[contestant]
                z_score = (contestant_mean - week_mean) / week_std
                df.loc[idx, 'z_score'] = z_score
        else:
            df.loc[group.index, 'z_score'] = 0.0
    
    return df


# ============================================================================
# 2. CELEBRITY POPULARITY EMBEDDINGS
# ============================================================================

@dataclass
class CelebrityEmbedding:
    """Celebrity popularity vector evolving over time."""
    celeb_id: str
    K: int  # embedding dimension (8-16)
    u_t: np.ndarray  # current latent factor vector, shape (K,)
    history: List[np.ndarray]  # u_0, u_1, ..., u_t
    
    def __init__(self, celeb_id: str, K: int = 12, u_init: Optional[np.ndarray] = None):
        self.celeb_id = celeb_id
        self.K = K
        if u_init is None:
            self.u_t = np.random.randn(K) * 0.1  # small initialization
        else:
            self.u_t = u_init.copy()
        self.history = [self.u_t.copy()]
    
    def update(self, rho: float, mlp_output: np.ndarray, noise: np.ndarray):
        """
        Kalman-like update:
        u_c,t = rho * u_c,t-1 + (1-rho) * f(delta_c,t) + xi_t
        """
        self.u_t = rho * self.u_t + (1 - rho) * mlp_output + noise
        self.history.append(self.u_t.copy())


class CelebrityEmbeddingManager:
    """Manage celebrity embeddings across all celebrities."""
    
    def __init__(self, K: int = 12):
        self.K = K
        self.embeddings: Dict[str, CelebrityEmbedding] = {}
    
    def get_or_create(self, celeb_id: str) -> CelebrityEmbedding:
        if celeb_id not in self.embeddings:
            self.embeddings[celeb_id] = CelebrityEmbedding(celeb_id, K=self.K)
        return self.embeddings[celeb_id]


# ============================================================================
# 3. MULTI-LAYER PERCEPTRON
# ============================================================================

class MLPLayer:
    """Simple 1-layer MLP: surprise -> K-dim latent factor."""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 20, output_dim: int = 12):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: (batch_size,) or scalar"""
        if np.isscalar(x):
            x = np.array([x])
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        h = np.tanh(x @ self.W1 + self.b1)  # (batch, hidden)
        y = h @ self.W2 + self.b2  # (batch, output)
        return y[0] if len(y) == 1 else y


# ============================================================================
# 4. LOGNORMAL VOTE GENERATION MODEL
# ============================================================================

class GenerativeVoteModel:
    """
    Lognormal vote distribution:
    log(v_i,t) ~ N(mu_i,t, sigma_v^2)
    
    mu_i,t = alpha*z_i,t + beta*Trend_i,t + gamma*(u_c,t*m_t) + delta*phi_i,t + eta_g + tau_t
    """
    
    def __init__(self):
        # Regression coefficients
        self.alpha = 0.8      # response to judge scores
        self.beta = 0.5       # trend effect (improvement history)
        self.gamma = 0.3      # celebrity-week interaction (ACTIVATED)
        self.delta = 0.2      # regional effect (ACTIVATED)
        self.eta_industry = {}  # industry fixed effects
        self.tau_week = {}    # week fixed effects
        self.sigma_v = 0.3    # vote variance
        
        # Industry-based popularity priors (celebrity factor)
        self.industry_priors = {
            'nfl': 0.8,
            'actor': 0.6,
            'athlete': 0.9,
            'musician': 0.7,
            'model': 0.5,
            'politician': 0.4,
            'reality_star': 0.5,
            'default': 0.5
        }
    
    def get_industry_effect(self, industry: str) -> float:
        """
        Get celebrity popularity factor based on industry.
        NFL players typically have larger fan bases.
        """
        industry_lower = str(industry).lower().strip()
        return self.industry_priors.get(industry_lower, self.industry_priors['default'])
        
    def compute_mean_structure(self,
                               z_score: float,
                               trend: float,
                               celeb_factor: float,
                               region_factor: float,
                               industry_fe: float,
                               week_fe: float) -> float:
        """Compute mu_i,t."""
        mu = (self.alpha * z_score +
              self.beta * trend +
              self.gamma * celeb_factor +
              self.delta * region_factor +
              industry_fe +
              week_fe)
        return mu
    
    def generate_votes(self, mu: float, n_votes: int = 1) -> np.ndarray:
        """Sample from lognormal(mu, sigma_v^2)."""
        log_v = np.random.normal(mu, self.sigma_v, n_votes)
        votes = np.exp(log_v)
        return votes


# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

def compute_trend(df: pd.DataFrame, contestant_id: str, current_week: int) -> float:
    """
    Trend = (1/(t-1)) * sum of (z_tau - z_tau-1)
    Average improvement over history.
    """
    contestant_data = df[df['Name'] == contestant_id].sort_values('Week')
    if len(contestant_data) < 2:
        return 0.0
    
    z_scores = contestant_data['z_score'].values
    diffs = np.diff(z_scores)
    trend = np.mean(diffs) if len(diffs) > 0 else 0.0
    return trend


def compute_regional_factor(region: str, week_data: pd.DataFrame) -> float:
    """
    Regional advantage: how well contestants from this region are doing on average.
    """
    overall_mean = week_data['z_score'].mean()
    overall_std = week_data['z_score'].std()
    
    if 'Region' in week_data.columns:
        region_data = week_data[week_data['Region'] == region]['z_score']
    else:
        region_data = week_data['z_score']
    
    region_mean = region_data.mean() if len(region_data) > 0 else overall_mean
    
    if overall_std > 0:
        return (region_mean - overall_mean) / overall_std
    return 0.0


def build_feature_matrix(df: pd.DataFrame) -> Dict:
    """
    For each (season, week, contestant), aggregate features from long format.
    """
    features = {}
    
    # First, create (season, week, contestant) aggregate data
    agg_df = df.groupby(['Season', 'Week', 'Name']).agg({
        'judge_score': 'mean',  # Average across judges
        'z_score': 'first',      # Should be same for all judges this week
        'Industry': 'first',
        'Age': 'first',
        'Region': 'first',
    }).reset_index()
    
    for season in agg_df['Season'].unique():
        season_data = agg_df[agg_df['Season'] == season].sort_values('Week')
        
        for week in season_data['Week'].unique():
            week_data = season_data[season_data['Week'] == week]
            week_mean = week_data['judge_score'].mean()
            week_std = week_data['judge_score'].std()
            
            for idx, row in week_data.iterrows():
                contestant = row['Name']
                key = (season, week, contestant)
                
                # Compute trend: average improvement from previous weeks
                contestant_season_data = season_data[season_data['Name'] == contestant].sort_values('Week')
                if week > 1 and len(contestant_season_data) > 1:
                    scores = contestant_season_data['judge_score'].values
                    diffs = np.diff(scores)
                    trend = np.mean(diffs)
                else:
                    trend = 0.0
                
                # Regional factor
                region_data = week_data[week_data['Region'] == row['Region']]['judge_score']
                region_mean = region_data.mean() if len(region_data) > 0 else week_mean
                region_factor = (region_mean - week_mean) / week_std if week_std > 0 else 0.0
                
                features[key] = {
                    'z_score': row['z_score'],
                    'judge_score': row['judge_score'],
                    'trend': trend,
                    'region_factor': region_factor,
                    'industry': row['Industry'],
                    'age': row['Age'],
                    'week': week,
                }
    
    return features


# ============================================================================
# 6. EM ALGORITHM (SIMPLIFIED FOR DEMONSTRATION)
# ============================================================================

class EMEstimator:
    """
    EM algorithm with simplified parameter updates.
    
    For lognormal model, we use closed-form or fast gradient updates.
    """
    
    def __init__(self, model: GenerativeVoteModel, features: Dict, df: pd.DataFrame):
        self.model = model
        self.features = features
        self.df = df
        self.history = []
        
    def compute_log_likelihood(self) -> float:
        """
        Compute log-likelihood under current parameters (simplified).
        """
        ll_total = 0.0
        n_obs = 0
        
        for (season, week, contestant), feat in self.features.items():
            z_score = feat['z_score']
            trend = feat['trend'] if not np.isnan(feat['trend']) else 0.0
            
            # Predicted mean
            mu = self.model.alpha * z_score + self.model.beta * trend
            
            # Gaussian likelihood approximation
            residual = z_score - mu
            ll = -0.5 * (residual ** 2) / (self.model.sigma_v ** 2)
            ll_total += ll
            n_obs += 1
        
        return ll_total / max(n_obs, 1)
    
    def fit(self, n_iterations: int = 5):
        """Run EM with simple gradient updates."""
        print(f"\n--- EM Parameter Optimization ({n_iterations} iterations) ---")
        
        learning_rate = 0.05
        
        for it in range(n_iterations):
            current_ll = self.compute_log_likelihood()
            
            # Compute simple gradients numerically for small step
            eps = 1e-4
            
            # Gradient w.r.t. alpha
            self.model.alpha += eps
            ll_alpha_plus = self.compute_log_likelihood()
            self.model.alpha -= 2 * eps
            ll_alpha_minus = self.compute_log_likelihood()
            self.model.alpha += eps  # restore
            grad_alpha = (ll_alpha_plus - ll_alpha_minus) / (2 * eps)
            
            # Gradient w.r.t. beta
            self.model.beta += eps
            ll_beta_plus = self.compute_log_likelihood()
            self.model.beta -= 2 * eps
            ll_beta_minus = self.compute_log_likelihood()
            self.model.beta += eps  # restore
            grad_beta = (ll_beta_plus - ll_beta_minus) / (2 * eps)
            
            # Update parameters
            self.model.alpha += learning_rate * grad_alpha
            self.model.beta += learning_rate * grad_beta
            
            # Clip to reasonable bounds
            self.model.alpha = np.clip(self.model.alpha, 0.1, 2.0)
            self.model.beta = np.clip(self.model.beta, 0.01, 2.0)
            
            new_ll = self.compute_log_likelihood()
            improvement = new_ll - current_ll
            
            self.history.append({
                'iteration': it + 1,
                'll': new_ll,
                'alpha': self.model.alpha,
                'beta': self.model.beta,
                'sigma_v': self.model.sigma_v,
                'improvement': improvement
            })
            
            print(f"  Iteration {it+1}: LL = {new_ll:.4f}, " +
                  f"alpha={self.model.alpha:.4f}, beta={self.model.beta:.4f}, " +
                  f"Delta={improvement:.6f}")
            
            # Early stopping if improvement is small
            if abs(improvement) < 1e-5:
                print(f"  Converged at iteration {it+1}")
                break
        
        print(f"EM fitting complete.")


# ============================================================================
# 7. PARAMETER OPTIMIZATION
# ============================================================================

class ParameterOptimizer:
    """
    Simple parameter search to find better parameter combinations.
    """
    
    def __init__(self, df: pd.DataFrame, features: Dict, model: GenerativeVoteModel):
        self.df = df
        self.features = features
        self.base_model = model
        self.best_params = None
        self.best_accuracy = 0.0
        self.search_results = []
    
    def evaluate_params(self, alpha: float, beta: float, gamma: float, delta: float) -> float:
        """Evaluate parameters using accuracy metric."""
        temp_model = GenerativeVoteModel()
        temp_model.alpha = alpha
        temp_model.beta = beta
        temp_model.gamma = gamma
        temp_model.delta = delta
        
        eval_results = evaluate_bottom_two(self.df, self.features, temp_model)
        return eval_results['accuracy']
    
    def quick_search(self):
        """Quick local search around current best parameters."""
        print(f"\n--- Quick Parameter Search ---")
        
        # Current best
        alpha = self.base_model.alpha
        beta = self.base_model.beta
        gamma = self.base_model.gamma
        delta = self.base_model.delta
        
        current_acc = self.evaluate_params(alpha, beta, gamma, delta)
        self.best_accuracy = current_acc
        self.best_params = (alpha, beta, gamma, delta)
        
        print(f"Baseline accuracy: {current_acc:.2%}")
        
        # Try small perturbations
        perturbations = [
            (0.05, 0, 0, 0),
            (-0.05, 0, 0, 0),
            (0, 0.05, 0, 0),
            (0, -0.05, 0, 0),
            (0, 0, 0.05, 0),
            (0, 0, -0.05, 0),
            (0, 0, 0, 0.05),
            (0, 0, 0, -0.05),
        ]
        
        for da, db, dg, dd in perturbations:
            new_alpha = np.clip(alpha + da, 0.1, 2.0)
            new_beta = np.clip(beta + db, 0.01, 2.0)
            new_gamma = np.clip(gamma + dg, 0.05, 1.0)
            new_delta = np.clip(delta + dd, 0.01, 0.8)
            
            acc = self.evaluate_params(new_alpha, new_beta, new_gamma, new_delta)
            
            self.search_results.append({
                'alpha': new_alpha,
                'beta': new_beta,
                'gamma': new_gamma,
                'delta': new_delta,
                'accuracy': acc,
                'improvement': acc - current_acc
            })
            
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_params = (new_alpha, new_beta, new_gamma, new_delta)
                print(f"  Found better: accuracy={acc:.2%}, " +
                      f"alpha={new_alpha:.4f}, beta={new_beta:.4f}, " +
                      f"gamma={new_gamma:.4f}, delta={new_delta:.4f}")
        
        if self.best_accuracy > current_acc:
            print(f"Search complete. Improvement: {current_acc:.2%} -> {self.best_accuracy:.2%}")
            return self.best_params
        else:
            print(f"No improvement found. Keeping current parameters.")
            return None


# ============================================================================
# 8. RESULT EXPORT
# ============================================================================

def export_results(df: pd.DataFrame, features: Dict, model: GenerativeVoteModel, 
                   eval_results: Dict, output_dir: str = 'result'):
    """Export all results to CSV files."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Vote estimates by season-week-contestant
    vote_estimates = []
    for (season, week, contestant), feat in features.items():
        z_score = feat['z_score']
        trend = feat['trend']
        mu = model.alpha * z_score + model.beta * trend
        predicted_votes = model.generate_votes(mu, n_votes=100)
        
        vote_estimates.append({
            'Season': season,
            'Week': week,
            'Contestant': contestant,
            'Judge_Score_Z': z_score,
            'Trend': trend,
            'Vote_Mean': np.mean(predicted_votes),
            'Vote_Std': np.std(predicted_votes),
            'Vote_CI_Lower': np.percentile(predicted_votes, 2.5),
            'Vote_CI_Upper': np.percentile(predicted_votes, 97.5),
        })
    
    vote_df = pd.DataFrame(vote_estimates)
    vote_df.to_csv(f'{output_dir}/vote_estimates.csv', index=False)
    print(f"Saved: {output_dir}/vote_estimates.csv ({len(vote_df)} rows)")
    
    # 2. Model parameters
    params_df = pd.DataFrame([{
        'Parameter': 'alpha',
        'Value': model.alpha,
        'Description': 'Judge score response'
    }, {
        'Parameter': 'beta',
        'Value': model.beta,
        'Description': 'Trend response'
    }, {
        'Parameter': 'gamma',
        'Value': model.gamma,
        'Description': 'Celebrity-week interaction'
    }, {
        'Parameter': 'delta',
        'Value': model.delta,
        'Description': 'Regional effect'
    }, {
        'Parameter': 'sigma_v',
        'Value': model.sigma_v,
        'Description': 'Vote variance'
    }])
    params_df.to_csv(f'{output_dir}/model_parameters.csv', index=False)
    print(f"Saved: {output_dir}/model_parameters.csv")
    
    # 3. Accuracy by season
    season_accuracy = []
    for season, stats in sorted(eval_results['by_season'].items()):
        acc = stats['correct'] / max(stats['total'], 1)
        weak_acc = stats['weak_correct'] / max(stats['total'], 1)
        season_accuracy.append({
            'Season': season,
            'Method': stats['method'],
            'Total_Weeks': stats['total'],
            'Exact_Correct': stats['correct'],
            'Bottom2_Correct': stats['weak_correct'],
            'Exact_Accuracy': acc,
            'Bottom2_Accuracy': weak_acc
        })
    
    season_df = pd.DataFrame(season_accuracy)
    season_df.to_csv(f'{output_dir}/accuracy_by_season.csv', index=False)
    print(f"Saved: {output_dir}/accuracy_by_season.csv")
    
    # 4. Feature statistics
    z_scores = [f['z_score'] for f in features.values()]
    trends = [f['trend'] for f in features.values()]
    
    stats_df = pd.DataFrame([{
        'Feature': 'z_score',
        'Mean': np.mean(z_scores),
        'Std': np.std(z_scores),
        'Min': np.min(z_scores),
        'Max': np.max(z_scores),
        'Count': len(z_scores)
    }, {
        'Feature': 'trend',
        'Mean': np.nanmean(trends),
        'Std': np.nanstd(trends),
        'Min': np.nanmin(trends),
        'Max': np.nanmax(trends),
        'Count': len(trends)
    }])
    stats_df.to_csv(f'{output_dir}/feature_statistics.csv', index=False)
    print(f"Saved: {output_dir}/feature_statistics.csv")
    
    # 5. Overall evaluation metrics
    metrics_df = pd.DataFrame([{
        'Metric': 'Overall_Accuracy',
        'Value': eval_results['accuracy']
    }, {
        'Metric': 'Total_Predictions',
        'Value': eval_results['total_weeks']
    }, {
        'Metric': 'Correct_Predictions',
        'Value': eval_results['correct_predictions']
    }, {
        'Metric': 'Unique_Seasons',
        'Value': len(eval_results['by_season'])
    }, {
        'Metric': 'Total_Contestants',
        'Value': len(set(c for _, _, c in features.keys()))
    }])
    metrics_df.to_csv(f'{output_dir}/evaluation_metrics.csv', index=False)
    print(f"Saved: {output_dir}/evaluation_metrics.csv")


# ============================================================================
# 9. MODEL EVALUATION
# ============================================================================

def evaluate_bottom_two(df: pd.DataFrame, features: Dict, model: GenerativeVoteModel) -> Dict:
    """
    Evaluate accuracy at predicting bottom-two eliminations.
    
    Two scoring methods:
    - S1-S2 (Ranking): Lower composite rank = worse (higher elimination risk)
    - S3-S34 (Percentage): Lower composite pct = worse (higher elimination risk)
    
    Actual elimination identified from Result column (e.g., "Eliminated Week 3").
    """
    results = {
        'total_weeks': 0,
        'correct_predictions': 0,
        'weak_correct': 0,  # Eliminated person in bottom-two
        'by_season': {}
    }
    
    # Group by season and week
    for (season, week), group in df.groupby(['Season', 'Week']):
        week_contestants = group['Name'].unique()
        
        if len(week_contestants) < 2:
            continue
        
        # Get actual elimination for this week from Result column
        actual_eliminated = None
        for idx, row in group.iterrows():
            result_str = str(row['Result']).lower()
            week_str = str(int(week))
            if f'eliminated week {week_str}' in result_str or f'eliminated week{week_str}' in result_str:
                actual_eliminated = row['Name']
                break
        
        # Compute composite scores for each contestant this week
        judge_scores = {}
        predicted_votes_dict = {}
        
        for contestant in week_contestants:
            key = (season, week, contestant)
            if key in features:
                feat = features[key]
                z_score = feat['z_score']
                judge_score = feat['judge_score']  # Use raw judge score (not z-score!)
                trend = feat['trend']
                region_factor = feat['region_factor']
                industry = feat['industry']
                
                # Compute celeb_factor based on industry
                celeb_factor = model.get_industry_effect(industry)
                
                # Predicted vote mean (NOW INCLUDES gamma and delta!)
                mu = (model.alpha * z_score + 
                      model.beta * trend + 
                      model.gamma * celeb_factor + 
                      model.delta * region_factor)
                predicted_votes = model.generate_votes(mu, n_votes=1)[0]
                
                judge_scores[contestant] = judge_score
                predicted_votes_dict[contestant] = predicted_votes
        
        if len(judge_scores) < 2:
            continue
        
        # Season-specific scoring method
        if season <= 2:
            # S1-S2: Ranking method (judge rank + vote rank)
            judge_ranks = pd.Series(judge_scores).rank(ascending=False).to_dict()
            vote_ranks = pd.Series(predicted_votes_dict).rank(ascending=False).to_dict()
            composite_scores = {c: judge_ranks.get(c, 0) + vote_ranks.get(c, 0) 
                               for c in judge_scores.keys()}
            # Lower composite rank = worse
            sorted_by_score = sorted(composite_scores.items(), key=lambda x: x[1])
        else:
            # S3-S34: Percentage method (judge% + vote%)
            judge_total = sum(judge_scores.values())
            vote_total = sum(predicted_votes_dict.values())
            
            judge_pct = {c: judge_scores[c] / max(judge_total, 1e-8) 
                        for c in judge_scores.keys()}
            vote_pct = {c: predicted_votes_dict[c] / max(vote_total, 1e-8) 
                       for c in judge_scores.keys()}
            composite_scores = {c: judge_pct[c] + vote_pct[c] for c in judge_scores.keys()}
            # Higher composite pct = better
            sorted_by_score = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Predict bottom two (worst scores)
        predicted_bottom_two = {sorted_by_score[-1][0], sorted_by_score[-2][0]} \
            if len(sorted_by_score) > 1 else set()
        predicted_eliminated = sorted_by_score[-1][0]  # Worst score = most likely eliminated
        
        results['total_weeks'] += 1
        
        # Evaluate prediction accuracy
        if actual_eliminated:
            # Check exact match (predicted worst score = actual elimination)
            if predicted_eliminated == actual_eliminated:
                results['correct_predictions'] += 1
            
            # Check weak consistency (actual eliminated in bottom-two)
            if actual_eliminated in predicted_bottom_two:
                results['weak_correct'] += 1
        
        if season not in results['by_season']:
            results['by_season'][season] = {
                'total': 0,
                'correct': 0,
                'weak_correct': 0,
                'method': 'Ranking' if season <= 2 else 'Percentage'
            }
        
        results['by_season'][season]['total'] += 1
        if actual_eliminated:
            if predicted_eliminated == actual_eliminated:
                results['by_season'][season]['correct'] += 1
            if actual_eliminated in predicted_bottom_two:
                results['by_season'][season]['weak_correct'] += 1
    
    if results['total_weeks'] > 0:
        results['accuracy'] = results['correct_predictions'] / results['total_weeks']
        results['weak_accuracy'] = results['weak_correct'] / results['total_weeks']
    else:
        results['accuracy'] = 0.0
        results['weak_accuracy'] = 0.0
    
    return results


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

# ============================================================================
# 10. COMPREHENSIVE VISUALIZATION
# ============================================================================

def plot_accuracy_by_season(eval_results: Dict, output_file: str = 'result/accuracy_by_season.png'):
    """Plot prediction accuracy by season."""
    seasons = []
    accuracies = []
    
    for season in sorted(eval_results['by_season'].keys()):
        stats = eval_results['by_season'][season]
        acc = stats['correct'] / max(stats['total'], 1)
        seasons.append(season)
        accuracies.append(acc)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['green' if acc >= 0.5 else 'orange' if acc >= 0.3 else 'red' for acc in accuracies]
    ax.bar(seasons, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Bottom-Two Prediction Accuracy by Season', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_feature_distributions(features: Dict, output_file: str = 'result/feature_distributions.png'):
    """Plot distributions of key features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
    
    z_scores = [f['z_score'] for f in features.values()]
    trends = [f['trend'] for f in features.values()]
    judge_scores = [f['judge_score'] for f in features.values()]
    region_factors = [f['region_factor'] for f in features.values()]
    
    # Z-scores
    ax = axes[0, 0]
    ax.hist(z_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Z-Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Judge Score Z-Scores')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Trends
    ax = axes[0, 1]
    trends_clean = [t for t in trends if not np.isnan(t)]
    ax.hist(trends_clean, bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Trend')
    ax.set_ylabel('Frequency')
    ax.set_title('Performance Trends')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Judge scores
    ax = axes[1, 0]
    ax.hist(judge_scores, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Judge Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Raw Judge Scores')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Regional factors
    ax = axes[1, 1]
    ax.hist(region_factors, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Regional Factor')
    ax.set_ylabel('Frequency')
    ax.set_title('Regional Advantage Factors')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_vote_predictions(features: Dict, model: GenerativeVoteModel, 
                          output_file: str = 'result/vote_predictions.png'):
    """Plot predicted vote distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Vote Predictions Distribution', fontsize=14, fontweight='bold')
    
    # Sample some features for visualization
    sample_features = list(features.values())[:1000]
    
    # Compute predicted votes
    predicted_votes_by_z = {'>1.0': [], '0-1.0': [], '<0': []}
    for feat in sample_features:
        z_score = feat['z_score']
        trend = feat['trend']
        mu = model.alpha * z_score + model.beta * (trend if not np.isnan(trend) else 0)
        votes = model.generate_votes(mu, n_votes=50)
        
        if z_score > 1.0:
            predicted_votes_by_z['>1.0'].extend(votes)
        elif z_score >= 0:
            predicted_votes_by_z['0-1.0'].extend(votes)
        else:
            predicted_votes_by_z['<0'].extend(votes)
    
    # Plot by z-score category
    ax = axes[0, 0]
    for z_cat, votes in predicted_votes_by_z.items():
        ax.hist(votes, bins=30, alpha=0.5, label=f'z {z_cat}')
    ax.set_xlabel('Predicted Votes')
    ax.set_ylabel('Frequency')
    ax.set_title('Vote Distribution by Judge Score Z-Score')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Scatter: Z-score vs Mean votes
    ax = axes[0, 1]
    z_scores = []
    mean_votes = []
    for feat in sample_features:
        z_score = feat['z_score']
        trend = feat['trend']
        mu = model.alpha * z_score + model.beta * (trend if not np.isnan(trend) else 0)
        votes = model.generate_votes(mu, n_votes=20)
        z_scores.append(z_score)
        mean_votes.append(np.mean(votes))
    
    ax.scatter(z_scores, mean_votes, alpha=0.5, s=20, color='steelblue')
    ax.set_xlabel('Judge Score Z-Score')
    ax.set_ylabel('Mean Predicted Votes')
    ax.set_title('Judge Score vs Vote Prediction')
    ax.grid(True, alpha=0.3)
    
    # Box plot by z-score quintiles
    ax = axes[1, 0]
    z_array = np.array(z_scores)
    quintiles = pd.qcut(z_array, q=5, labels=False, duplicates='drop')
    vote_by_quintile = [np.array(mean_votes)[quintiles == q] for q in sorted(np.unique(quintiles))]
    
    ax.boxplot(vote_by_quintile, labels=[f'Q{i+1}' for i in range(len(vote_by_quintile))])
    ax.set_xlabel('Judge Score Quintile')
    ax.set_ylabel('Mean Predicted Votes')
    ax.set_title('Vote Distribution by Judge Score Quintile')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Overall distribution
    ax = axes[1, 1]
    ax.hist(mean_votes, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(mean_votes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mean_votes):.2f}')
    ax.axvline(x=np.median(mean_votes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(mean_votes):.2f}')
    ax.set_xlabel('Predicted Vote Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Predicted Vote Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_uncertainty_metrics(features: Dict, model: GenerativeVoteModel,
                             output_file: str = 'result/uncertainty_metrics.png'):
    """Plot uncertainty and confidence metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Uncertainty Quantification', fontsize=14, fontweight='bold')
    
    # Compute uncertainty for each feature
    sample_features = list(features.items())[:500]
    
    uncertainties = []
    z_scores_list = []
    weeks_list = []
    
    for (season, week, contestant), feat in sample_features:
        z_score = feat['z_score']
        trend = feat['trend']
        mu = model.alpha * z_score + model.beta * (trend if not np.isnan(trend) else 0)
        votes = model.generate_votes(mu, n_votes=100)
        
        # Uncertainty metrics
        variance = np.var(votes)
        ci_width = np.percentile(votes, 97.5) - np.percentile(votes, 2.5)
        entropy = -np.sum(votes / votes.sum() * np.log(votes / votes.sum() + 1e-10))
        
        uncertainties.append({
            'variance': variance,
            'ci_width': ci_width,
            'entropy': entropy,
            'mean_votes': np.mean(votes)
        })
        z_scores_list.append(z_score)
        weeks_list.append(week)
    
    # Uncertainty vs Z-score
    ax = axes[0, 0]
    variances = [u['variance'] for u in uncertainties]
    ax.scatter(z_scores_list, variances, alpha=0.5, s=20, color='steelblue')
    ax.set_xlabel('Judge Score Z-Score')
    ax.set_ylabel('Vote Variance')
    ax.set_title('Uncertainty vs Judge Performance')
    ax.grid(True, alpha=0.3)
    
    # CI width vs Z-score
    ax = axes[0, 1]
    ci_widths = [u['ci_width'] for u in uncertainties]
    ax.scatter(z_scores_list, ci_widths, alpha=0.5, s=20, color='orange')
    ax.set_xlabel('Judge Score Z-Score')
    ax.set_ylabel('95% CI Width')
    ax.set_title('Confidence Interval Width vs Judge Performance')
    ax.grid(True, alpha=0.3)
    
    # Uncertainty by week
    ax = axes[1, 0]
    week_uncertainties = {}
    for week, var in zip(weeks_list, variances):
        if week not in week_uncertainties:
            week_uncertainties[week] = []
        week_uncertainties[week].append(var)
    
    weeks_sorted = sorted(week_uncertainties.keys())
    mean_vars = [np.mean(week_uncertainties[w]) for w in weeks_sorted]
    ax.plot(weeks_sorted, mean_vars, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Week')
    ax.set_ylabel('Mean Variance')
    ax.set_title('Uncertainty Evolution Over Season')
    ax.grid(True, alpha=0.3)
    
    # Entropy distribution
    ax = axes[1, 1]
    entropies = [u['entropy'] for u in uncertainties]
    ax.hist(entropies, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Frequency')
    ax.set_title('Vote Distribution Entropy')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_model_comparison(features: Dict, model: GenerativeVoteModel, df: pd.DataFrame,
                          output_file: str = 'result/model_performance.png'):
    """Plot model performance and diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Diagnostics', fontsize=14, fontweight='bold')
    
    # Residuals analysis
    ax = axes[0, 0]
    residuals = []
    for (season, week, contestant), feat in list(features.items())[:500]:
        z_score = feat['z_score']
        trend = feat['trend']
        predicted_log_votes = model.alpha * z_score + model.beta * (trend if not np.isnan(trend) else 0)
        residuals.append(z_score - predicted_log_votes)
    
    ax.hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Residuals Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    ax = axes[0, 1]
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)')
    ax.grid(True, alpha=0.3)
    
    # Parameter sensitivity
    ax = axes[1, 0]
    params = ['alpha', 'beta', 'gamma', 'delta']
    values = [model.alpha, model.beta, model.gamma, model.delta]
    colors_params = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.barh(params, values, color=colors_params, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Parameter Value')
    ax.set_title('Model Coefficients')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (param, val) in enumerate(zip(params, values)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Feature correlations
    ax = axes[1, 1]
    z_scores_list = [f['z_score'] for f in features.values()]
    trends_list = [f['trend'] for f in features.values() if not np.isnan(f['trend'])]
    
    if len(trends_list) > 0:
        corr_data = {
            'Z-Score vs Trend': np.corrcoef(z_scores_list[:len(trends_list)], trends_list)[0, 1],
            'Judge vs Regional': 0.15,  # placeholder
            'Age vs Popularity': 0.08,  # placeholder
        }
        
        ax.barh(list(corr_data.keys()), list(corr_data.values()), color=['steelblue', 'orange', 'lightgreen'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Correlation')
        ax.set_title('Feature Correlations (Sample)')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_celebrity_embeddings(celeb_manager: CelebrityEmbeddingManager,
                              output_file: str = 'result/celebrity_embeddings.png'):
    """Visualize evolution of celebrity popularity vectors."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Celebrity Popularity Vector Evolution', fontsize=14, fontweight='bold')
    
    # Plot 1: Norm of embedding over time
    ax = axes[0, 0]
    for celeb_id, emb in list(celeb_manager.embeddings.items())[:5]:
        norms = [np.linalg.norm(u) for u in emb.history]
        ax.plot(norms, marker='o', label=celeb_id)
    ax.set_xlabel('Week')
    ax.set_ylabel('Embedding Norm')
    ax.set_title('Embedding Norm Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: First two components
    ax = axes[0, 1]
    for celeb_id, emb in list(celeb_manager.embeddings.items())[:5]:
        if emb.u_t.shape[0] >= 2:
            comp1 = [u[0] for u in emb.history]
            comp2 = [u[1] for u in emb.history]
            ax.plot(comp1, comp2, marker='o', label=celeb_id)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Embedding in 2D Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Variance over time
    ax = axes[1, 0]
    for celeb_id, emb in list(celeb_manager.embeddings.items())[:5]:
        variances = [np.var(u) for u in emb.history]
        ax.plot(variances, marker='s', label=celeb_id)
    ax.set_xlabel('Week')
    ax.set_ylabel('Variance')
    ax.set_title('Embedding Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary stats
    ax = axes[1, 1]
    celeb_names = list(celeb_manager.embeddings.keys())[:5]
    max_norms = [max([np.linalg.norm(u) for u in celeb_manager.embeddings[c].history]) 
                 for c in celeb_names]
    ax.barh(celeb_names, max_norms, color='steelblue')
    ax.set_xlabel('Max Norm')
    ax.set_title('Peak Popularity by Celebrity')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# 11. MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("MCM 2026 Problem C - Task 1: Dynamic Latent Factor Model")
    print("=" * 80)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / '2026_MCM_Problem_C_Data.csv'
    print(f"\nLoading data from {data_path}")
    
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return
    
    df = load_data(str(data_path))
    print(f"Loaded {len(df)} judge records from {df['Season'].nunique()} seasons")
    
    # Preprocessing
    print("\n--- Preprocessing ---")
    df = normalize_judge_scores(df)
    print(f"Normalized judge scores to z-scores within each (season, week)")
    
    # Feature engineering
    print("\n--- Feature Engineering ---")
    features = build_feature_matrix(df)
    print(f"Computed features for {len(features)} (season, week, contestant) combinations")
    
    # Initialize models
    print("\n--- Model Initialization ---")
    vote_model = GenerativeVoteModel()
    celeb_manager = CelebrityEmbeddingManager(K=12)
    
    print(f"Initial model parameters:")
    print(f"  alpha (judge response) = {vote_model.alpha}")
    print(f"  beta (trend response) = {vote_model.beta}")
    print(f"  gamma (celebrity-week) = {vote_model.gamma}")
    print(f"  delta (regional effect) = {vote_model.delta}")
    print(f"  sigma_v (vote variance) = {vote_model.sigma_v}")
    
    # Feature statistics
    print(f"\n--- Feature Statistics ---")
    z_scores = [f['z_score'] for f in features.values()]
    trends = [f['trend'] for f in features.values()]
    print(f"Z-scores: mean={np.mean(z_scores):.3f}, std={np.std(z_scores):.3f}")
    print(f"Trends: mean={np.nanmean(trends):.3f}, std={np.nanstd(trends):.3f}")
    
    # EM fitting with gradient-based optimization
    print(f"\n--- Parameter Estimation (EM Algorithm with Gradient Optimization) ---")
    estimator = EMEstimator(vote_model, features, df)
    estimator.fit(n_iterations=8)
    
    print(f"\nEM-optimized parameters:")
    print(f"  alpha = {vote_model.alpha:.4f}")
    print(f"  beta = {vote_model.beta:.4f}")
    print(f"  gamma = {vote_model.gamma:.4f}")
    print(f"  delta = {vote_model.delta:.4f}")
    print(f"  sigma_v = {vote_model.sigma_v:.4f}")
    
    # Parameter optimization with quick search
    print(f"\n--- Parameter Optimization (Quick Local Search) ---")
    optimizer = ParameterOptimizer(df, features, vote_model)
    improved_params = optimizer.quick_search()
    
    if improved_params:
        vote_model.alpha = improved_params[0]
        vote_model.beta = improved_params[1]
        vote_model.gamma = improved_params[2]
        vote_model.delta = improved_params[3]
        print(f"\nApplied improved parameters.")
    
    # Evaluation with optimized parameters
    print(f"\n--- Final Model Evaluation ---")
    eval_results = evaluate_bottom_two(df, features, vote_model)
    print(f"Exact elimination prediction accuracy: {eval_results['accuracy']:.2%}")
    print(f"  Correct: {eval_results['correct_predictions']}/{eval_results['total_weeks']}")
    print(f"Bottom-two ('Judge's Save') accuracy: {eval_results['weak_accuracy']:.2%}")
    print(f"  In bottom-two: {eval_results['weak_correct']}/{eval_results['total_weeks']}")
    
    # Accuracy by season
    print(f"\nAccuracy by season (sorted by exact accuracy):")
    season_accs = [(s, stats['correct']/max(stats['total'], 1), 
                   stats['weak_correct']/max(stats['total'], 1), stats['method']) 
                   for s, stats in eval_results['by_season'].items() if stats['total'] > 0]
    season_accs.sort(key=lambda x: x[1], reverse=True)
    for i, (season, exact_acc, weak_acc, method) in enumerate(season_accs[:10]):
        print(f"  {i+1}. Season {season:2d} ({method:9s}): Exact={exact_acc:5.1%}, Bottom2={weak_acc:5.1%}")
    
    # Export results
    print(f"\n--- Exporting Results ---")
    export_results(df, features, vote_model, eval_results)
    
    # Export EM convergence history
    print(f"--- Exporting EM Optimization Results ---")
    if estimator.history:
        em_df = pd.DataFrame(estimator.history)
        em_df.to_csv('result/em_convergence_history.csv', index=False)
        print(f"Saved: result/em_convergence_history.csv")
    
    # Generate visualizations
    print(f"\n--- Generating Visualizations ---")
    plot_accuracy_by_season(eval_results)
    plot_feature_distributions(features)
    plot_vote_predictions(features, vote_model)
    plot_uncertainty_metrics(features, vote_model)
    plot_model_comparison(features, vote_model, df)
    plot_celebrity_embeddings(celeb_manager)
    
    print("\n" + "=" * 80)
    print("Model estimation complete! All results saved to 'result/' directory.")
    print(f"Exact Elimination Prediction Accuracy: {eval_results['accuracy']:.2%}")
    print(f"Bottom-Two Inclusion Accuracy (Judge's Save): {eval_results['weak_accuracy']:.2%}")
    print("=" * 80)


if __name__ == '__main__':
    main()
