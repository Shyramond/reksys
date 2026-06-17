from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else 0.0


class RecommenderEvaluator:
    def __init__(self, full_feature_matrix, item_popularity: Optional[Dict[int, float]] = None, item_index: Optional[Sequence[int]] = None):
        self.X = full_feature_matrix
        self.item_popularity = item_popularity or {}
        self.item_index = list(item_index) if item_index is not None else None
        total = sum(self.item_popularity.values()) if self.item_popularity else 0
        self._total_interactions = total if total > 0 else 0

    def precision_at_k(self, recs: Sequence[int], relevant: set, k: int) -> float:
        if not recs:
            return 0.0
        rec_k = recs[:k]
        return len(set(rec_k) & relevant) / len(rec_k)

    def recall_at_k(self, recs: Sequence[int], relevant: set, k: int) -> float:
        if not relevant:
            return 0.0
        rec_k = recs[:k]
        return len(set(rec_k) & relevant) / len(relevant)

    def map_at_k(self, recs: Sequence[int], relevant: set, k: int) -> float:
        if not relevant:
            return 0.0
        score = 0.0
        hits = 0
        for i, item in enumerate(recs[:k], start=1):
            if item in relevant:
                hits += 1
                score += hits / i
        return score / min(len(relevant), k) if hits > 0 else 0.0

    def ndcg_at_k(self, recs: Sequence[int], relevant: set, k: int) -> float:
        dcg = 0.0
        for rank, item in enumerate(recs[:k], start=1):
            if item in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        return dcg / ideal if ideal > 0 else 0.0

    def mrr_at_k(self, recs: Sequence[int], relevant: set, k: int) -> float:
        for rank, item in enumerate(recs[:k], start=1):
            if item in relevant:
                return 1.0 / rank
        return 0.0

    def catalog_coverage(self, all_recommendations: Sequence[Sequence[int]], n_items: Optional[int] = None) -> float:
        unique = set()
        for rec in all_recommendations:
            unique.update(rec)
        if n_items is None:
            try:
                n_items = int(self.X.shape[0])
            except Exception:
                n_items = len(self.item_popularity) if self.item_popularity else len(unique)
        return len(unique) / n_items if n_items > 0 else 0.0

    def novelty_self_information(self, recs: Sequence[int]) -> float:
        vals = []
        V = len(self.item_popularity) if self.item_popularity else 0
        total = self._total_interactions if self._total_interactions > 0 else 0
        for item in recs:
            count = self.item_popularity.get(item, 0)
            if total > 0 and V > 0:
                p = (count + 1) / (total + V)
            else:
                p = 1.0 / (V + 1) if V > 0 else 0.0
            if p > 0:
                vals.append(-math.log2(p))
        return float(np.mean(vals)) if vals else 0.0

    def ild(self, recs: Sequence[int]) -> float:
        k = len(recs)
        if k <= 1:
            return 0.0
        rows = []
        for item in recs:
            idx = item if self.item_index is None else (self.item_index.index(item) if item in self.item_index else None)
            if idx is None:
                continue
            row = self.X[idx]
            try:
                arr = row.toarray().ravel()
            except Exception:
                arr = np.asarray(row).ravel()
            rows.append(arr)
        if len(rows) <= 1:
            return 0.0
        M = np.vstack(rows)
        dists = distance.pdist(M, metric='cosine')
        return float(np.mean(dists))

    def personalization(self, all_recommendations: Sequence[Sequence[int]]) -> float:
        lists = [list(r) for r in all_recommendations]
        n = len(lists)
        if n <= 1:
            return 0.0
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                A = set(lists[i])
                B = set(lists[j])
                inter = len(A & B)
                union = len(A | B)
                jacc = inter / union if union > 0 else 0.0
                sims.append(jacc)
        mean_jacc = float(np.mean(sims)) if sims else 0.0
        return 1.0 - mean_jacc

    def evaluate(self,
                 test_queries: Sequence,
                 recommendation_engine: Callable[[int, int], Sequence[int]],
                 k: int = 10,
                 build_relevant_fn: Optional[Callable[[int], set]] = None) -> Dict:
        per_query = []
        all_recs = []
        for q in test_queries:
            if isinstance(q, tuple) and len(q) >= 2:
                seed_idx, relevant = q[0], q[1]
            else:
                seed_idx = q
                relevant = build_relevant_fn(seed_idx) if build_relevant_fn is not None else set()
            recs = list(recommendation_engine(seed_idx, k))
            all_recs.append(recs)
            prec = self.precision_at_k(recs, relevant, k)
            rec = self.recall_at_k(recs, relevant, k)
            mapk = self.map_at_k(recs, relevant, k)
            ndcg = self.ndcg_at_k(recs, relevant, k)
            mrr = self.mrr_at_k(recs, relevant, k)
            nov = self.novelty_self_information(recs)
            ild_v = self.ild(recs)
            per_query.append({
                'seed': seed_idx,
                'recs': recs,
                'precision': prec,
                'recall': rec,
                'map': mapk,
                'ndcg': ndcg,
                'mrr': mrr,
                'novelty': nov,
                'ild': ild_v,
            })

        metrics = {
            'precision': _safe_mean([p['precision'] for p in per_query]),
            'recall': _safe_mean([p['recall'] for p in per_query]),
            'map': _safe_mean([p['map'] for p in per_query]),
            'ndcg': _safe_mean([p['ndcg'] for p in per_query]),
            'mrr': _safe_mean([p['mrr'] for p in per_query]),
            'novelty': _safe_mean([p['novelty'] for p in per_query]),
            'ild': _safe_mean([p['ild'] for p in per_query]),
            'catalog_coverage': self.catalog_coverage(all_recs),
            'personalization': self.personalization(all_recs),
            'per_query': per_query,
            'all_recs': all_recs,
        }
        return metrics

    def write_report(self, metrics: Dict, out_path: str = 'offline_evaluation_report.txt') -> None:
        lines = []
        lines.append('Offline Evaluation Report')
        lines.append('=' * 40)
        lines.append('')
        lines.append('Summary Metrics:')
        lines.append(f"  Precision@K : {metrics['precision']:.4f}")
        lines.append(f"  Recall@K    : {metrics['recall']:.4f}")
        lines.append(f"  MAP@K       : {metrics['map']:.4f}")
        lines.append(f"  NDCG@K      : {metrics['ndcg']:.4f}")
        lines.append(f"  MRR@K       : {metrics['mrr']:.4f}")
        lines.append(f"  Novelty     : {metrics['novelty']:.4f}")
        lines.append(f"  ILD         : {metrics['ild']:.4f}")
        lines.append(f"  Coverage    : {metrics['catalog_coverage']:.4f}")
        lines.append(f"  Personaliz. : {metrics['personalization']:.4f}")
        lines.append('')
        lines.append('Analysis:')
        if metrics['ild'] < 0.1 and metrics['precision'] > 0.15:
            lines.append('  - Model achieves good accuracy but low diversity: trade-off observed.')
        else:
            lines.append('  - No strong evidence of sacrificing diversity for accuracy on these queries.')
        lines.append('')
        lines.append('Per-query sample (first 10):')
        lines.append('seed | precision | recall | map | ndcg | mrr | novelty | ild')
        for pq in metrics['per_query'][:10]:
            lines.append(f"{pq['seed']} | {pq['precision']:.4f} | {pq['recall']:.4f} | {pq['map']:.4f} | {pq['ndcg']:.4f} | {pq['mrr']:.4f} | {pq['novelty']:.4f} | {pq['ild']:.4f}")

        Path(out_path).write_text('\n'.join(lines), encoding='utf-8')

    def plot_long_tail(self, metrics: Dict, out_path: str = 'long_tail_coverage.png') -> None:
        if plt is None:
            return
        all_recs = metrics.get('all_recs', [])
        flat = [i for rec in all_recs for i in rec]
        if not flat:
            return
        counts = pd.Series(flat).value_counts().sort_values(ascending=False)
        cum_counts = counts.cumsum() / counts.sum()
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind='bar', ax=ax, alpha=0.7)
        ax2 = ax.twinx()
        cum_counts.plot(ax=ax2, color='red', linewidth=2)
        ax.set_xlabel('Item')
        ax.set_ylabel('Recommendation Count')
        ax2.set_ylabel('Cumulative Coverage')
        fig.tight_layout()
        fig.savefig(out_path)


def _example_usage():
    try:
        import importlib.util as iu, sys
        spec = iu.spec_from_file_location('sk', Path(__file__).resolve().parent / 'SVD+CB' / 'steam_knn_svd.py')
        sk = iu.module_from_spec(spec)
        spec.loader.exec_module(sk)
        df = pd.read_csv(Path(__file__).resolve().parent / 'SVD+CB' / 'steam_top_games_2026.csv')
        df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns = sk.build_feature_matrix(df)
        def engine(seed_idx, k=10):
            return sk.sequential_svd_then_refine_then_content(seed_idx, None, X_text, X_content, X_img, df, top_k=k)
        pop = {i: max(1, len(str(r.get('tags','')).split(','))) for i, r in df.iterrows()}
        evaluator = RecommenderEvaluator(full_feature_matrix=X_content, item_popularity=pop, item_index=list(df.index))
        candidate_indices = list(range(len(df)))[:200]
        queries = []
        for idx in candidate_indices:
            tags = [t.strip().lower() for t in str(df.loc[idx,'tags']).split(',') if t.strip()]
            if not tags:
                continue
            hidden = tags[0]
            relevant = set(df.index[df['tags'].str.lower().str.contains(rf"\b{re.escape(hidden)}\b", na=False)])
            relevant.discard(idx)
            queries.append((idx, relevant))
        metrics = evaluator.evaluate(queries, engine, k=10)
        evaluator.write_report(metrics, out_path=str(Path(__file__).resolve().parent / 'offline_evaluation_report.txt'))
        if plt is not None:
            evaluator.plot_long_tail(metrics, out_path=str(Path(__file__).resolve().parent / 'long_tail_coverage.png'))
    except Exception:
        pass


if __name__ == '__main__':
    _example_usage()
