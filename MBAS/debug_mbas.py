"""Debug MBAS to understand why Hybrid systems aren't selected."""

import importlib.util
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODULE_PATH = ROOT_DIR / "SVD+CB" / "steam_knn_svd.py"
DATA_PATH = ROOT_DIR / "SVD+CB" / "steam_top_games_2026.csv"

ALPHA = 1.0
TOP_K = 10
RANDOM_STATE = 42


class LinUCBBandit:
    def __init__(self, n_arms: int, d: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A = np.array([np.eye(d, dtype=np.float64) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, d), dtype=np.float64)

    def _arm_score(self, arm: int, x: np.ndarray, alpha: float) -> float:
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv.dot(self.b[arm])
        exploitation = float(theta.dot(x))
        exploration = float(alpha * np.sqrt(x.dot(A_inv.dot(x))))
        return exploitation + exploration

    def get_all_scores(self, context: np.ndarray, alpha: float | None = None) -> Dict[int, float]:
        """Get scores for all arms."""
        alpha = self.alpha if alpha is None else alpha
        context = np.asarray(context, dtype=np.float64)
        scores = {}
        for arm in range(self.n_arms):
            scores[arm] = self._arm_score(arm, context, alpha)
        return scores

    def select(self, context: np.ndarray, alpha: float | None = None) -> int:
        scores = self.get_all_scores(context, alpha)
        return max(scores, key=scores.get)

    def update(self, arm: int, reward: float, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


def load_steam_module():
    if not MODULE_PATH.exists():
        raise FileNotFoundError(f"Steam module not found at {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("steam_knn_svd_mod", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if hasattr(module, "report_file"):
        try:
            module.report_file.close()
        except Exception:
            pass
    return module


def main():
    # Load module and data
    module = load_steam_module()
    df = pd.read_csv(DATA_PATH)
    
    # Build features (same as MBAS)
    df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns = \
        module.build_feature_matrix(df)
    
    X_text_mean = np.mean(module.TruncatedSVD(n_components=20, random_state=RANDOM_STATE).fit_transform(X_text), axis=0)
    
    # Build arms (simplified, just to test)
    arms = {}
    
    # Arm 1: Content KNN
    content_knn = module.NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    content_knn.fit(X_content)
    def content_recommend(seed_idx, k=TOP_K):
        distances, indices = content_knn.kneighbors(X_content[seed_idx], n_neighbors=k + 1)
        return [i for i in indices[0] if i != seed_idx][:k]
    arms["Content_KNN"] = content_recommend
    
    # Arm 2: Hybrid SVD-first
    X_full = module.sparse.hstack([X_text, X_extra], format="csr")
    X_full = module.normalize(X_full, norm="l2", axis=1)
    
    svd_best = module.TruncatedSVD(n_components=20, random_state=RANDOM_STATE)
    X_text_best = svd_best.fit_transform(X_text)
    X_text_best_norm = module.normalize(X_text_best, norm="l2", axis=1)
    svd_knn_best = module.NearestNeighbors(n_neighbors=TOP_K * 10 + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    svd_knn_best.fit(X_text_best_norm)

    def hybrid_svd_recommend(seed_idx, k=TOP_K):
        return module.sequential_svd_then_refine_then_content(seed_idx, svd_knn_best, X_text_best_norm, X_full, X_img, df, top_k=k)
    arms["Hybrid_SVD_First"] = hybrid_svd_recommend
    
    # Test on first 20 iterations
    n_arms = len(arms)
    arm_names = list(arms.keys())
    bandit = LinUCBBandit(n_arms=n_arms, d=X_text_mean.shape[0], alpha=ALPHA)
    
    rng = np.random.default_rng(RANDOM_STATE)
    seed_indices = df.index.tolist()
    
    print("First 30 iterations (arm selection & scores):\n")
    print(f"{'Iter':<4} {'Arm':<20} {'Content Score':<15} {'Hybrid Score':<15} {'Top-1 Hit':<10} {'Reward':<8}")
    print("-" * 80)
    
    for iteration in range(1, 31):
        seed_idx = int(rng.choice(seed_indices))
        
        # Get scores
        context = X_text_mean
        scores = bandit.get_all_scores(context, alpha=ALPHA)
        
        # Select arm
        selected_arm_idx = max(range(len(scores)), key=lambda i: scores[i])
        selected_arm_name = arm_names[selected_arm_idx]
        selected_engine = arms[selected_arm_name]
        
        # Get recommendations
        try:
            recs = selected_engine(seed_idx, k=TOP_K)
            top1 = recs[0] if recs else None
            error = None
        except Exception as e:
            top1 = None
            error = str(e)[:30]
        
        # Calculate reward
        def get_hidden_tag(df_row):
            tags = [tag.strip().lower() for tag in str(df_row["tags"]).split(",") if tag.strip()]
            return tags[0] if tags else None
        
        hidden_tag = get_hidden_tag(df.iloc[seed_idx])
        reward = 0
        if top1 is not None and hidden_tag:
            top1_tags = {tag.strip().lower() for tag in str(df.loc[top1, "tags"]).split(",") if tag.strip()}
            reward = 1 if hidden_tag in top1_tags else 0
        
        # Update bandit
        bandit.update(selected_arm_idx, reward, context)
        
        # Print
        print(f"{iteration:<4} {selected_arm_name:<20} {scores[0]:<15.6f} {scores[1]:<15.6f} {str(top1):<10} {reward:<8}", end="")
        if error:
            print(f" ERROR: {error}")
        else:
            print()
    
    print("\nFinal arm scores after 30 iterations:")
    final_scores = bandit.get_all_scores(X_text_mean, alpha=ALPHA)
    for i, name in enumerate(arm_names):
        print(f"  {name:<20} {final_scores[i]:.6f}")


if __name__ == "__main__":
    main()
