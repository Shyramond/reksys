import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODULE_PATH = ROOT_DIR / "SVD+CB" / "steam_knn_svd.py"
REPORT_PATH = SCRIPT_DIR / "bandit_experiment_report.txt"
BANDIT_LOG_PATH = SCRIPT_DIR / "bandit_simulation_output.txt"
DATA_PATH = ROOT_DIR / "SVD+CB" / "steam_top_games_2026.csv"

ALPHA = 1.0
SVD_COMPONENTS = 20
ITERATIONS = 1000
TOP_K = 10
TOP_N_CANDIDATES = 50
REPORT_BATCH = 100
RANDOM_STATE = 42


class LinUCBBandit:
    def __init__(self, n_arms: int, d: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A = np.array([np.eye(d, dtype=np.float64) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, d), dtype=np.float64)

    def _theta(self, arm: int) -> np.ndarray:
        A_inv = np.linalg.inv(self.A[arm])
        return A_inv.dot(self.b[arm])

    def _arm_score(self, arm: int, x: np.ndarray, alpha: float) -> float:
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv.dot(self.b[arm])
        exploitation = float(theta.dot(x))
        exploration = float(alpha * np.sqrt(x.dot(A_inv.dot(x))))
        return exploitation + exploration

    def recommend(self, candidates: list[int], seed_context: np.ndarray, candidate_contexts: np.ndarray, alpha: float | None = None, top_k: int = TOP_K) -> list[int]:
        alpha = self.alpha if alpha is None else alpha
        if isinstance(seed_context, list):
            seed_context = np.asarray(seed_context, dtype=np.float64)
        if candidate_contexts.ndim == 1:
            candidate_contexts = candidate_contexts.reshape(1, -1)
        seed_context = seed_context.astype(np.float64)
        x_mat = candidate_contexts * seed_context[None, :]
        scores = []
        for idx, arm in enumerate(candidates):
            x = x_mat[idx]
            score = self._arm_score(arm, x, alpha)
            scores.append((arm, score))
        ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
        return [arm for arm, _ in ranked[:top_k]]

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


def get_hybrid_candidates(module, seed_idx: int, svd_knn, X_svd: np.ndarray, full_matrix, top_n: int = TOP_N_CANDIDATES) -> list[int]:
    candidates = module.get_svd_candidates(seed_idx, svd_knn, X_svd, top_n=top_n)
    refined = module.refine_candidates_by_similarity(seed_idx, candidates, full_matrix, top_n=top_n)
    return refined


def get_hybrid_recommendations(module, seed_idx: int, svd_knn, X_svd: np.ndarray, X_content, X_img, df, top_k: int = TOP_K) -> list[int]:
    return module.sequential_svd_then_refine_then_content(seed_idx, svd_knn, X_svd, X_content, X_img, df, top_k=top_k)


def get_static_hybrid_rank(module, seed_idx: int, content_knn, X_content, svd_knn, X_svd, full_matrix, X_img, df, top_k: int = TOP_K) -> list[int]:
    return module.sequential_svd_then_refine_then_content(seed_idx, svd_knn, X_svd, full_matrix, X_img, df, top_k=top_k)


def hide_token_from_text(text: str, token: str) -> str:
    if not text or not token:
        return text
    token = token.lower()
    return re.sub(rf"\b{re.escape(token)}\b", "", text.lower())


def precision_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    if not recs:
        return 0.0
    return len(set(recs[:k]) & relevant) / len(recs[:k])


def load_data(module):
    df = pd.read_csv(DATA_PATH)
    df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns = module.build_feature_matrix(df)
    return df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns


def build_seed_indices(df: pd.DataFrame) -> list[int]:
    valid_tags = [
        tag.strip().lower()
        for tags in df["tags"].fillna("")
        for tag in tags.split(",")
        if tag.strip()
    ]
    candidate_indices = [
        idx for idx, row in df.iterrows()
        if len([t for t in str(row["tags"]).split(",") if t.strip()]) >= 1
    ]
    return candidate_indices


def get_hidden_tag(df: pd.DataFrame, seed_idx: int) -> str | None:
    tags = [tag.strip().lower() for tag in str(df.loc[seed_idx, "tags"]).split(",") if tag.strip()]
    return tags[0] if tags else None


def build_relevant_set(df: pd.DataFrame, hidden_tag: str, exclude_idx: int) -> set[int]:
    pattern = rf"\b{re.escape(hidden_tag)}\b"
    relevant = set(df.index[df["tags"].str.lower().str.contains(pattern, na=False)])
    relevant.discard(exclude_idx)
    return relevant


def run_bandit_simulation(module, df, X_svd, svd_knn, X_content, X_img):
    n_arms = len(df)
    bandit = LinUCBBandit(n_arms=n_arms, d=X_svd.shape[1], alpha=ALPHA)
    rng = np.random.default_rng(RANDOM_STATE)
    seed_indices = build_seed_indices(df)
    if not seed_indices:
        raise ValueError("No valid seed games with tags available for bandit simulation.")

    cumulative_reward = 0
    exploration_count = 0
    bandit_hit_count = 0
    static_hit_count = 0
    bandit_precision_sum = 0.0
    static_precision_sum = 0.0
    history = []

    for iteration in range(1, ITERATIONS + 1):
        seed_idx = int(rng.choice(seed_indices))
        hidden_tag = get_hidden_tag(df, seed_idx)
        if hidden_tag is None:
            continue
        relevant_set = build_relevant_set(df, hidden_tag, seed_idx)
        if not relevant_set:
            continue

        seed_context = X_svd[seed_idx]
        candidates = get_hybrid_candidates(module, seed_idx, svd_knn, X_svd, X_content, top_n=TOP_N_CANDIDATES)
        if not candidates:
            continue

        candidate_contexts = X_svd[candidates]
        static_top10 = get_static_hybrid_rank(module, seed_idx, None, X_content, svd_knn, X_svd, X_content, X_img, df, top_k=TOP_K)
        static_top1 = static_top10[0] if static_top10 else None

        bandit_top10 = bandit.recommend(candidates, seed_context, candidate_contexts, alpha=ALPHA, top_k=TOP_K)
        bandit_top1 = bandit_top10[0] if bandit_top10 else None

        if static_top1 is not None and bandit_top1 is not None and static_top1 != bandit_top1:
            exploration_count += 1

        bandit_tags = {tag.strip().lower() for tag in str(df.loc[bandit_top1, "tags"]).split(",") if tag.strip()} if bandit_top1 is not None else set()
        static_tags = {tag.strip().lower() for tag in str(df.loc[static_top1, "tags"]).split(",") if tag.strip()} if static_top1 is not None else set()
        bandit_reward = 1 if hidden_tag in bandit_tags else 0
        static_reward = 1 if hidden_tag in static_tags else 0

        if bandit_top1 is not None:
            bandit.update(bandit_top1, bandit_reward, seed_context * X_svd[bandit_top1])

        cumulative_reward += bandit_reward
        bandit_hit_count += bandit_reward
        static_hit_count += static_reward
        bandit_precision_sum += precision_at_k(bandit_top10, relevant_set, TOP_K)
        static_precision_sum += precision_at_k(static_top10, relevant_set, TOP_K)

        if iteration % REPORT_BATCH == 0:
            history.append({
                "iteration": iteration,
                "cumulative_reward": cumulative_reward,
                "bandit_precision": bandit_precision_sum / iteration,
                "static_precision": static_precision_sum / iteration,
                "exploration_rate": exploration_count / iteration,
                "bandit_top1_hit_rate": bandit_hit_count / iteration,
                "static_top1_hit_rate": static_hit_count / iteration,
            })

    return {
        "iterations": iteration,
        "cumulative_reward": cumulative_reward,
        "bandit_precision": bandit_precision_sum / iteration,
        "static_precision": static_precision_sum / iteration,
        "exploration_rate": exploration_count / iteration,
        "bandit_top1_hit_rate": bandit_hit_count / iteration,
        "static_top1_hit_rate": static_hit_count / iteration,
        "history": history,
    }


def write_report(metrics: dict):
    lines = [
        "LinUCB Bandit Re-ranker Experiment Report",
        "=" * 48,
        "",
        "Hyperparameters:",
        f"  - alpha: {ALPHA}",
        f"  - SVD dimensions: {SVD_COMPONENTS}",
        f"  - iterations: {ITERATIONS}",
        f"  - stage-2 candidate pool: {TOP_N_CANDIDATES}",
        f"  - final top-K: {TOP_K}",
        "",
        "Final Metrics:",
        f"  - cumulative reward: {metrics['cumulative_reward']}",
        f"  - average bandit Precision@{TOP_K}: {metrics['bandit_precision']:.4f}",
        f"  - average static Precision@{TOP_K}: {metrics['static_precision']:.4f}",
        f"  - exploration rate: {metrics['exploration_rate']:.4f}",
        f"  - bandit top-1 hit rate: {metrics['bandit_top1_hit_rate']:.4f}",
        f"  - static top-1 hit rate: {metrics['static_top1_hit_rate']:.4f}",
        "",
        "Key Findings:",
        "  - LinUCB reranked the top-50 hybrid candidates using arm-specific contextual scoring.",
        "  - The experiment compares the static hybrid model against the hybrid+LinUCB re-ranker.",
        "  - Exploration is measured as cases where the bandit selected a different top-1 game than the static hybrid model.",
        "  - The bandit is rewarded when the selected recommendation shares the seed game's hidden tag.",
        "",
    ]
    lines += ["Performance summary every 100 iterations:", "Iteration | CumulativeReward | BanditPrec | StaticPrec | ExplorationRate | BanditTop1HR | StaticTop1HR", "-" * 96]
    for row in metrics["history"]:
        lines.append(
            f"{row['iteration']:9d} | {row['cumulative_reward']:16d} | {row['bandit_precision']:.4f} | {row['static_precision']:.4f} | {row['exploration_rate']:.4f} | {row['bandit_top1_hit_rate']:.4f} | {row['static_top1_hit_rate']:.4f}"
        )
    lines.append("")
    improvement = metrics['bandit_top1_hit_rate'] - metrics['static_top1_hit_rate']
    lines.append(f"Hit rate improvement (bandit top-1 vs static top-1): {improvement:.4f}")
    lines.append("")
    lines.append("Observations:")
    lines.append("  - If the bandit hit rate exceeds the static hit rate, the re-ranker improves stage-3 recommendations.")
    lines.append("  - A non-zero exploration rate shows the bandit actively tries alternatives beyond the hybrid model's first choice.")
    lines.append("")
    Path(REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")


def write_bandit_log(metrics: dict):
    header = [
        "Bandit Simulation Output",
        "========================",
        "",
        f"Iterations: {metrics['iterations']}",
        f"Alpha: {ALPHA}",
        f"SVD dimensions: {SVD_COMPONENTS}",
        f"Top candidates pool: {TOP_N_CANDIDATES}",
        f"Final Top-K: {TOP_K}",
        "",
        "Iteration, CumulativeReward, BanditPrecision, StaticPrecision, ExplorationRate, BanditTop1HitRate, StaticTop1HitRate",
    ]
    rows = [
        ", ".join([
            str(row["iteration"]),
            str(row["cumulative_reward"]),
            f"{row['bandit_precision']:.4f}",
            f"{row['static_precision']:.4f}",
            f"{row['exploration_rate']:.4f}",
            f"{row['bandit_top1_hit_rate']:.4f}",
            f"{row['static_top1_hit_rate']:.4f}",
        ])
        for row in metrics["history"]
    ]
    Path(BANDIT_LOG_PATH).write_text("\n".join(header + rows), encoding="utf-8")


def main():
    module = load_steam_module()
    df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns = load_data(module)
    svd_model, X_svd, svd_knn = module.build_svd_index(X_text, n_components=SVD_COMPONENTS, n_neighbors=TOP_N_CANDIDATES + 1)
    metrics = run_bandit_simulation(module, df, X_svd, svd_knn, X_content, X_img)
    write_report(metrics)
    write_bandit_log(metrics)
    print(f"Bandit experiment completed. Report written to {REPORT_PATH}")
    print(f"Detailed bandit output written to {BANDIT_LOG_PATH}")


if __name__ == "__main__":
    main()
