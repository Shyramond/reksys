import importlib.util
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Callable, Tuple, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODULE_PATH = ROOT_DIR / "SVD+CB" / "steam_knn_svd.py"
REPORT_PATH = SCRIPT_DIR / "mbas_experiment_report.txt"
LOG_PATH = SCRIPT_DIR / "mbas_simulation_output.txt"
DATA_PATH = ROOT_DIR / "SVD+CB" / "steam_top_games_2026.csv"

ALPHA = 1.0
ITERATIONS = 1000
TOP_K = 10
RANDOM_STATE = 42
REPORT_BATCH = 100


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

    def select(self, context: np.ndarray, alpha: float | None = None) -> int:
        """Select best arm based on context."""
        alpha = self.alpha if alpha is None else alpha
        context = np.asarray(context, dtype=np.float64)
        best_arm, best_score = 0, float('-inf')
        for arm in range(self.n_arms):
            score = self._arm_score(arm, context, alpha)
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

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


def build_recommender_arms(module, df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns):
    arms = {}

    # Arm 0: Raw KNN
    raw_knn = module.NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    raw_knn.fit(X_base)

    def raw_recommend(seed_idx, k=TOP_K):
        distances, indices = raw_knn.kneighbors(X_base[seed_idx], n_neighbors=k + 1)
        return [i for i in indices[0] if i != seed_idx][:k]

    arms["Raw_KNN"] = raw_recommend

    # Arm 1: Content-based KNN
    content_knn = module.NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    content_knn.fit(X_content)

    def content_recommend(seed_idx, k=TOP_K):
        distances, indices = content_knn.kneighbors(X_content[seed_idx], n_neighbors=k + 1)
        return [i for i in indices[0] if i != seed_idx][:k]

    arms["Content_KNN"] = content_recommend

    # Arms 2-4: SVD KNN with different dimensions
    for dim in [20, 50, 100]:
        svd = module.TruncatedSVD(n_components=dim, random_state=RANDOM_STATE)
        X_text_reduced = svd.fit_transform(X_text)
        X_reduced = np.hstack([X_text_reduced, X_extra.toarray()])
        X_reduced = module.normalize(X_reduced, norm="l2", axis=1)

        svd_knn = module.NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        svd_knn.fit(X_reduced)

        def make_svd_recommend(knn, svd_m, extra):
            def recommend(seed_idx, k=TOP_K):
                query_text_reduced = svd_m.transform(vectorizer.transform([module.build_item_text(df.iloc[seed_idx])]))
                query_extra = np.hstack([
                    numeric.iloc[[seed_idx]].values,
                    df[bool_columns].astype(float).iloc[[seed_idx]].values,
                    pd.get_dummies(df["estimated_owners"].iloc[[seed_idx]]).reindex(columns=owner_columns, fill_value=0).values
                ])
                query_reduced = np.hstack([query_text_reduced, query_extra])
                query_reduced = module.normalize(query_reduced, norm="l2", axis=1)
                distances, indices = knn.kneighbors(query_reduced[0], n_neighbors=k + 1)
                return [i for i in indices[0] if i != seed_idx][:k]
            return recommend

        arms[f"SVD_{dim}"] = make_svd_recommend(svd_knn, svd, X_reduced)

    X_full = module.sparse.hstack([X_text, X_extra], format="csr")
    X_full = module.normalize(X_full, norm="l2", axis=1)

    # Arm 5: Hybrid SVD-first (should use full matrix for refinement, not just content)
    svd_best = module.TruncatedSVD(n_components=20, random_state=RANDOM_STATE)
    X_text_best = svd_best.fit_transform(X_text)
    X_text_best_norm = module.normalize(X_text_best, norm="l2", axis=1)
    svd_knn_best = module.NearestNeighbors(n_neighbors=TOP_K * 10 + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    svd_knn_best.fit(X_text_best_norm)

    def hybrid_svd_recommend(seed_idx, k=TOP_K):
        # Pass X_full (text + numeric + bool + owner) for refinement, not X_content
        return module.sequential_svd_then_refine_then_content(seed_idx, svd_knn_best, X_text_best_norm, X_full, X_img, df, top_k=k)

    arms["Hybrid_SVD_First"] = hybrid_svd_recommend

    # Arm 6: Hybrid content-first (uses full matrix for refinement as well)
    content_knn2 = module.NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    content_knn2.fit(X_content)

    def hybrid_content_recommend(seed_idx, k=TOP_K):
        distances, indices = content_knn2.kneighbors(X_content[seed_idx], n_neighbors=50 + 1)
        candidates = [i for i in indices[0] if i != seed_idx][:50]
        # Use X_full for refinement, not X_content
        refined = module.refine_candidates_by_similarity(seed_idx, candidates, X_full, top_n=30)
        personalized = module.personalize_candidates(seed_idx, refined, X_img, df, top_k=k)
        return personalized

    arms["Hybrid_Content_First"] = hybrid_content_recommend

    return arms


def get_hidden_tag(df: pd.DataFrame, seed_idx: int) -> Optional[str]:
    """Extract first tag from game."""
    tags = [tag.strip().lower() for tag in str(df.loc[seed_idx, "tags"]).split(",") if tag.strip()]
    return tags[0] if tags else None


def build_relevant_set(df: pd.DataFrame, hidden_tag: str, exclude_idx: int) -> set:
    """Build set of games with the given tag."""
    pattern = rf"\b{re.escape(hidden_tag)}\b"
    relevant = set(df.index[df["tags"].str.lower().str.contains(pattern, na=False)])
    relevant.discard(exclude_idx)
    return relevant


def run_mbas_simulation(arms: Dict[str, Callable], df, seed_indices, vectorizer, X_text, module) -> Dict:
    """Run MBAS: select best recommender via LinUCB across iterations.
    
    Args:
        arms: Dict of recommender systems
        df: DataFrame with games
        seed_indices: List of candidate game indices
        vectorizer: TfidfVectorizer fitted on corpus
        X_text: TF-IDF text matrix
        module: Loaded steam_knn_svd module
    """
    n_arms = len(arms)
    arm_names = list(arms.keys())
    # SVD context dimension is 20
    bandit = LinUCBBandit(n_arms=n_arms, d=20, alpha=ALPHA)

    rng = np.random.default_rng(RANDOM_STATE)
    cumulative_reward = 0
    per_arm_hits = defaultdict(int)
    per_arm_trials = defaultdict(int)
    history = []
    
    # Pre-fit SVD for computing context vectors
    svd_context = module.TruncatedSVD(n_components=20, random_state=RANDOM_STATE)
    svd_context.fit(X_text)

    for iteration in range(1, ITERATIONS + 1):
        seed_idx = int(rng.choice(seed_indices))
        hidden_tag = get_hidden_tag(df, seed_idx)
        if hidden_tag is None:
            continue

        relevant_set = build_relevant_set(df, hidden_tag, seed_idx)
        if not relevant_set:
            continue

        # context: SVD representation of the seed game itself (query-dependent)
        # This allows LinUCB to adapt based on different queries
        seed_text_tfidf = vectorizer.transform([module.build_item_text(df.iloc[seed_idx])])
        context = svd_context.transform(seed_text_tfidf)[0]
        
        selected_arm_idx = bandit.select(context, alpha=ALPHA)
        selected_arm_name = arm_names[selected_arm_idx]
        selected_engine = arms[selected_arm_name]

        try:
            recs = selected_engine(seed_idx, k=TOP_K)
            top1 = recs[0] if recs else None
        except Exception:
            top1 = None

        # reward: 1 if top-1 shares the hidden tag
        reward = 0
        if top1 is not None:
            top1_tags = {tag.strip().lower() for tag in str(df.loc[top1, "tags"]).split(",") if tag.strip()}
            reward = 1 if hidden_tag in top1_tags else 0

        # update bandit
        bandit.update(selected_arm_idx, reward, context)
        cumulative_reward += reward
        per_arm_hits[selected_arm_name] += reward
        per_arm_trials[selected_arm_name] += 1

        if iteration % REPORT_BATCH == 0:
            arm_rates = {name: per_arm_hits[name] / max(1, per_arm_trials[name]) for name in arm_names}
            history.append({
                "iteration": iteration,
                "cumulative_reward": cumulative_reward,
                "selected_arm": selected_arm_name,
                "arm_hit_rates": arm_rates.copy(),
            })

    return {
        "iterations": iteration,
        "cumulative_reward": cumulative_reward,
        "per_arm_hits": dict(per_arm_hits),
        "per_arm_trials": dict(per_arm_trials),
        "arm_names": arm_names,
        "history": history,
    }


def write_mbas_report(metrics: Dict, out_path: str = "mbas_experiment_report.txt") -> None:
    """Write MBAS results to human-readable report."""
    lines = []
    lines.append("Multi-Bandit Arm Selection (MBAS) Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Hyperparameters:")
    lines.append(f"  - alpha: {ALPHA}")
    lines.append(f"  - iterations: {ITERATIONS}")
    lines.append(f"  - top-K: {TOP_K}")
    lines.append(f"  - number of arms: {len(metrics['arm_names'])}")
    lines.append("")
    lines.append("Arms (Recommender Systems):")
    for i, name in enumerate(metrics['arm_names']):
        lines.append(f"  {i}: {name}")
    lines.append("")
    lines.append("Final Results:")
    lines.append(f"  - Total Iterations: {metrics['iterations']}")
    lines.append(f"  - Cumulative Reward: {metrics['cumulative_reward']}")
    lines.append("")
    lines.append("Per-Arm Performance:")
    lines.append("  Arm Name                   | Hits | Trials | Hit Rate")
    lines.append("  " + "-" * 60)
    for arm_name in metrics['arm_names']:
        hits = metrics['per_arm_hits'].get(arm_name, 0)
        trials = metrics['per_arm_trials'].get(arm_name, 1)
        rate = hits / trials if trials > 0 else 0.0
        lines.append(f"  {arm_name:27s} | {hits:4d} | {trials:6d} | {rate:.4f}")
    lines.append("")
    lines.append("Key Findings:")
    best_arm = max(metrics['arm_names'], key=lambda x: metrics['per_arm_hits'].get(x, 0) / max(1, metrics['per_arm_trials'].get(x, 1)))
    best_rate = metrics['per_arm_hits'].get(best_arm, 0) / max(1, metrics['per_arm_trials'].get(best_arm, 1))
    lines.append(f"  - Best performing arm: {best_arm} (hit rate: {best_rate:.4f})")
    lines.append("  - LinUCB adaptively selected between different recommender systems.")
    lines.append("  - The bandit learned which system performs best across test queries.")
    lines.append("")
    lines.append("Sample History (every 100 iterations):")
    lines.append("Iteration | CumulativeReward | SelectedArm")
    for h in metrics['history'][:10]:
        lines.append(f"{h['iteration']:9d} | {h['cumulative_reward']:16d} | {h['selected_arm']}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_mbas_log(metrics: Dict, out_path: str = "mbas_simulation_output.txt") -> None:
    """Write detailed MBAS simulation log."""
    lines = []
    lines.append("MBAS Simulation Output")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Total Iterations: {metrics['iterations']}")
    lines.append(f"Cumulative Reward: {metrics['cumulative_reward']}")
    lines.append("")
    lines.append("Arm, Hits, Trials, HitRate")
    for arm_name in metrics['arm_names']:
        hits = metrics['per_arm_hits'].get(arm_name, 0)
        trials = metrics['per_arm_trials'].get(arm_name, 1)
        rate = hits / trials if trials > 0 else 0.0
        lines.append(f"{arm_name}, {hits}, {trials}, {rate:.4f}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    # Load module and data
    module = load_steam_module()
    df = pd.read_csv(DATA_PATH)
    df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns = module.build_feature_matrix(df)

    # Build recommender arms
    arms = build_recommender_arms(module, df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns)

    # Prepare seed indices
    tag_counts = Counter(
        tag.strip().lower()
        for tags in df["tags"].fillna("")
        for tag in tags.split(",")
        if tag.strip()
    )
    valid_tags = {tag for tag, count in tag_counts.items() if 20 <= count <= 150}
    candidate_indices = [
        idx for idx, row in df.iterrows()
        if len([t for t in str(row["tags"]).split(",") if t.strip().lower() in valid_tags]) >= 2
    ]
    if len(candidate_indices) < 100:
        candidate_indices = [i for i in range(len(df))]

    # Run MBAS with query-dependent context (not global mean)
    metrics = run_mbas_simulation(arms, df, candidate_indices, vectorizer, X_text, module)

    # Write reports
    write_mbas_report(metrics, out_path=str(REPORT_PATH))
    write_mbas_log(metrics, out_path=str(LOG_PATH))

    print(f"MBAS experiment completed.")
    print(f"Report written to {REPORT_PATH}")
    print(f"Log written to {LOG_PATH}")


if __name__ == "__main__":
    main()
