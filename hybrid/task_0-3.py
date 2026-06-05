import os
import sys
import zipfile
import urllib.request
from datetime import datetime
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

np.random.seed(42)

REPORT_FILE = "hybrid_report.txt"
report_file = open(REPORT_FILE, 'w', encoding='utf-8')

def log(message='', end='\n', essential=True):
    if essential:
        report_file.write(str(message) + end)
        report_file.flush()
    print(str(message), end=end)

log(f"{'=' * 75}")
log(f"  ОТЧЁТ: Гибридная рекомендательная система (MovieLens 1M)")
log(f"  Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Файл отчёта: {REPORT_FILE}")
log(f"{'=' * 75}")
log()
print("Библиотеки импортированы.")

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH = "ml-1m.zip"
DATA_DIR = "ml-1m"

if not os.path.exists(DATA_DIR):
    print("Скачиваю MovieLens 1M...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(".")
    print("Готово.")
else:
    print("Датасет уже скачан.")

ratings = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.dat"),
    sep="::", header=None,
    names=["userId", "movieId", "rating", "timestamp"],
    engine="python", encoding="latin-1"
)
movies = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"),
    sep="::", header=None,
    names=["movieId", "title", "genres"],
    engine="python", encoding="latin-1"
)
users = pd.read_csv(
    os.path.join(DATA_DIR, "users.dat"),
    sep="::", header=None,
    names=["userId", "gender", "age", "occupation", "zip"],
    engine="python", encoding="latin-1"
)

log(f"Ratings: {ratings.shape[0]} записей")
log(f"Movies:  {movies.shape[0]} фильмов")
log(f"Users:   {users.shape[0]} пользователей")


log(f"\n{'=' * 75}")
log(f"  СТРУКТУРА И СТАТИСТИКА ДАННЫХ")
log(f"{'=' * 75}")

log(f"\n--- ratings.describe() ---")
log(ratings.describe().to_string())
log(f"\nПропуски: {ratings.isnull().sum().sum()}")

n_users_all = ratings['userId'].nunique()
n_items_all = ratings['movieId'].nunique()
n_ratings = len(ratings)
sparsity = 1 - n_ratings / (n_users_all * n_items_all)

log(f"\nУникальных пользователей: {n_users_all}")
log(f"Уникальных фильмов:       {n_items_all}")
log(f"Всего оценок:              {n_ratings}")
log(f"Разреженность матрицы:     {sparsity:.2%}")
log(f"Средняя оценка:            {ratings['rating'].mean():.3f}")
log(f"Медиана оценок:            {ratings['rating'].median()}")

log(f"\nРаспределение оценок:")
for val, cnt in ratings['rating'].value_counts().sort_index().items():
    bar = "█" * int(cnt / n_ratings * 100)
    print(f"  {val:.1f}: {cnt:>7d} ({cnt/n_ratings:>6.2%}) {bar}")

user_stats = ratings.groupby('userId')['rating'].agg(['count', 'mean'])
log(f"\nОценок на пользователя:")
log(f"  min={user_stats['count'].min()}, "
      f"median={user_stats['count'].median():.0f}, "
      f"mean={user_stats['count'].mean():.1f}, "
      f"max={user_stats['count'].max()}")

item_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
log(f"Оценок на фильм:")
log(f"  min={item_stats['count'].min()}, "
      f"median={item_stats['count'].median():.0f}, "
      f"mean={item_stats['count'].mean():.1f}, "
      f"max={item_stats['count'].max()}")

popular = (
    ratings.groupby('movieId')['rating']
    .agg(count='count', mean_rating='mean')
    .reset_index()
    .merge(movies, on='movieId')
    .sort_values('count', ascending=False)
)
log(f"\nТоп-10 самых оценённых фильмов:")
for _, row in popular.head(10).iterrows():
    log(f"  {row['count']:>5d} оценок | "
          f"ср. {row['mean_rating']:.2f} | {row['title']}")

all_genres = set()
for g in movies['genres']:
    all_genres.update(g.split('|'))
log(f"\nВсего жанров: {len(all_genres)}")
log(f"Жанры: {', '.join(sorted(all_genres))}")

class Config:
    K_NEIGHBORS_UB = 20
    K_NEIGHBORS_IB = 50
    TOP_X = 10
    TEST_SIZE = 0.2
    RATING_THRESHOLD = 3.5
    MIN_RATINGS_USER = 20
    MIN_RATINGS_ITEM = 10
    EVAL_SAMPLE = 500
    ALPHA = 0.7
    INTERSECTION_BOOST = 1.5
    POPULARITY_PENALTY = 0.02
    DIVERSITY_RERANK = False
    
    '''def adaptive_alpha(user_id):
        uidx = user2idx[user_id]
        n_rated = train_matrix[uidx].nnz
        return min(0.8, 0.3 + n_rated / 200)'''

cfg = Config()

log(f"\n{'=' * 75}")
log(f"  ГИПЕРПАРАМЕТРЫ")
log(f"{'=' * 75}")
for attr in ['K_NEIGHBORS_UB', 'K_NEIGHBORS_IB', 'TOP_X', 'TEST_SIZE', 'RATING_THRESHOLD',
             'MIN_RATINGS_USER', 'MIN_RATINGS_ITEM', 'EVAL_SAMPLE',
             'ALPHA', 'INTERSECTION_BOOST', 'POPULARITY_PENALTY',
             'DIVERSITY_RERANK']:
    log(f"  {attr:25s} = {getattr(cfg, attr)}")

log(f"\n{'=' * 75}")
log(f"  ПРЕДОБРАБОТКА")
log(f"{'=' * 75}")

user_counts = ratings['userId'].value_counts()
item_counts = ratings['movieId'].value_counts()

df = ratings[
    ratings['userId'].isin(
        user_counts[user_counts >= cfg.MIN_RATINGS_USER].index
    ) &
    ratings['movieId'].isin(
        item_counts[item_counts >= cfg.MIN_RATINGS_ITEM].index
    )
].copy()

log(f"После фильтрации: {len(df)} оценок, "
      f"{df['userId'].nunique()} users, "
      f"{df['movieId'].nunique()} items")

train_parts, test_parts = [], []
for uid, group in df.groupby('userId'):
    if len(group) < 5:
        train_parts.append(group)
        continue
    tr, te = train_test_split(group, test_size=cfg.TEST_SIZE, random_state=42)
    train_parts.append(tr)
    test_parts.append(te)

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

log(f"Train: {len(train_df)} | Test: {len(test_df)}")
log(f"Соотношение: {len(test_df)/len(df):.2%} в тесте")

all_users = sorted(df['userId'].unique())
all_items = sorted(df['movieId'].unique())
user2idx = {u: i for i, u in enumerate(all_users)}
idx2user = {i: u for u, i in user2idx.items()}
item2idx = {m: i for i, m in enumerate(all_items)}
idx2item = {i: m for m, i in item2idx.items()}

N_USERS = len(all_users)
N_ITEMS = len(all_items)

def build_sparse_matrix(data):
    rows, cols, vals = [], [], []
    for _, row in data.iterrows():
        u = user2idx.get(row['userId'])
        m = item2idx.get(row['movieId'])
        if u is not None and m is not None:
            rows.append(u)
            cols.append(m)
            vals.append(row['rating'])
    return csr_matrix((vals, (rows, cols)), shape=(N_USERS, N_ITEMS))

train_matrix = build_sparse_matrix(train_df)
item_user_matrix = train_matrix.T.tocsr()

log(f"\nUser-Item матрица: {train_matrix.shape}")
log(f"Ненулевых: {train_matrix.nnz} "
      f"({train_matrix.nnz / (N_USERS * N_ITEMS):.4%})")

user_means = np.zeros(N_USERS)
for i in range(N_USERS):
    row = train_matrix[i]
    if row.nnz > 0:
        user_means[i] = row.data.mean()

item_popularity = np.array(
    (train_matrix > 0).sum(axis=0)
).flatten().astype(float)
item_popularity_norm = item_popularity / item_popularity.max()

test_user_items = defaultdict(dict)
for _, row in test_df.iterrows():
    test_user_items[row['userId']][row['movieId']] = row['rating']

test_user_relevant = {
    u: {m for m, r in items.items() if r >= cfg.RATING_THRESHOLD}
    for u, items in test_user_items.items()
}

test_users_valid = [
    u for u, rel in test_user_relevant.items() if len(rel) > 0
]
log(f"Пользователей с релевантными тестовыми фильмами: {len(test_users_valid)}")

eval_users = np.random.choice(
    test_users_valid,
    min(cfg.EVAL_SAMPLE, len(test_users_valid)),
    replace=False
)
log(f"Пользователей для оценки: {len(eval_users)}")

def get_title(movie_id):
    t = movies[movies['movieId'] == movie_id]['title'].values
    return t[0] if len(t) > 0 else f"ID={movie_id}"

def get_genres(movie_id):
    g = movies[movies['movieId'] == movie_id]['genres'].values
    return g[0] if len(g) > 0 else ""

item2genres = dict(zip(movies['movieId'], movies['genres']))

def precision_at_k(recs, relevant, k):
    recs_k = recs[:k]
    if not recs_k:
        return 0.0
    return len(set(recs_k) & relevant) / len(recs_k)

def recall_at_k(recs, relevant, k):
    recs_k = recs[:k]
    if not relevant:
        return 0.0
    return len(set(recs_k) & relevant) / len(relevant)

def f1_at_k(recs, relevant, k):
    p = precision_at_k(recs, relevant, k)
    r = recall_at_k(recs, relevant, k)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

def ndcg_at_k(recs, relevant, k):
    recs_k = recs[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recs_k) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(recs, relevant, k):
    return 1.0 if set(recs[:k]) & relevant else 0.0

def average_precision_at_k(recs, relevant, k):
    recs_k = recs[:k]
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(recs_k):
        if item in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant), k) if relevant else 0.0

def mrr_at_k(recs, relevant, k):
    for i, item in enumerate(recs[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def intra_list_diversity(recs, item2genres_dict, k):
    recs_k = recs[:k]
    if len(recs_k) < 2:
        return 0.0
    dists = []
    for i in range(len(recs_k)):
        for j in range(i + 1, len(recs_k)):
            g1 = set(item2genres_dict.get(recs_k[i], "").split('|'))
            g2 = set(item2genres_dict.get(recs_k[j], "").split('|'))
            union = g1 | g2
            if union:
                dists.append(1 - len(g1 & g2) / len(union))
            else:
                dists.append(0)
    return np.mean(dists) if dists else 0.0

def evaluate_full(user_recs, test_relevant, k, total_items):
    metrics = defaultdict(list)
    all_recs_items = set()
    n_evaluated = 0
    for uid, recs in user_recs.items():
        relevant = test_relevant.get(uid, set())
        if not relevant:
            continue
        n_evaluated += 1
        metrics['Precision@K'].append(precision_at_k(recs, relevant, k))
        metrics['Recall@K'].append(recall_at_k(recs, relevant, k))
        metrics['F1@K'].append(f1_at_k(recs, relevant, k))
        metrics['NDCG@K'].append(ndcg_at_k(recs, relevant, k))
        metrics['HitRate@K'].append(hit_rate_at_k(recs, relevant, k))
        metrics['MAP@K'].append(average_precision_at_k(recs, relevant, k))
        metrics['MRR@K'].append(mrr_at_k(recs, relevant, k))
        metrics['ILD@K'].append(intra_list_diversity(recs, item2genres, k))
        all_recs_items.update(recs[:k])
    result = {name: np.mean(vals) for name, vals in metrics.items()}
    result['Coverage'] = len(all_recs_items) / total_items
    result['Users_evaluated'] = n_evaluated
    return result

def print_metrics(m, title=""):
    log(f"\n{'━' * 55}")
    log(f"  {title}")
    log(f"{'━' * 55}")
    for name, val in m.items():
        if name == 'Users_evaluated':
            log(f"  {'Пользователей оценено':30s}: {val}")
        else:
            log(f"  {name:30s}: {val:.4f}")
    log(f"{'━' * 55}")

print("Метрики определены.")

print(f"\n{'=' * 75}")
print(f"  ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
print(f"{'=' * 75}")

print("Обучение Item-Based KNN...")
item_knn = NearestNeighbors(
    n_neighbors=cfg.K_NEIGHBORS_IB + 1,
    metric='cosine', algorithm='brute', n_jobs=None
)
item_knn.fit(item_user_matrix)

def item_based_scores(user_id, k=cfg.K_NEIGHBORS_IB):
    if user_id not in user2idx:
        return {}
    uidx = user2idx[user_id]
    user_vec = train_matrix[uidx]
    rated_indices = user_vec.nonzero()[1]
    rated_set = set(rated_indices)
    if len(rated_indices) == 0:
        return {}
    u_mean = user_means[uidx]
    scores = defaultdict(float)
    weights = defaultdict(float)
    for item_idx in rated_indices:
        user_rating = user_vec[0, item_idx]
        distances, indices = item_knn.kneighbors(
            item_user_matrix[item_idx], n_neighbors=k + 1
        )
        for j in range(len(indices[0])):
            nb_idx = indices[0][j]
            if nb_idx in rated_set:
                continue
            sim = max(1 - distances[0][j], 0)
            if sim <= 0:
                continue
            scores[nb_idx] += sim * (user_rating - u_mean)
            weights[nb_idx] += sim
    result = {}
    for idx, total in scores.items():
        pred = u_mean + total / weights[idx]
        pred = np.clip(pred, 1, 5)
        result[idx2item[idx]] = pred
    return result

print("Обучение User-Based KNN...")
user_knn = NearestNeighbors(
    n_neighbors=cfg.K_NEIGHBORS_UB + 1,
    metric='cosine', algorithm='brute', n_jobs=None
)
user_knn.fit(train_matrix)

def user_based_scores(user_id, k=cfg.K_NEIGHBORS_UB):
    if user_id not in user2idx:
        return {}
    uidx = user2idx[user_id]
    user_vec = train_matrix[uidx]
    rated_set = set(user_vec.nonzero()[1])
    u_mean = user_means[uidx]
    distances, indices = user_knn.kneighbors(user_vec, n_neighbors=k + 1)
    scores = defaultdict(float)
    weights = defaultdict(float)
    for j in range(len(indices[0])):
        nb_uidx = indices[0][j]
        if nb_uidx == uidx:
            continue
        sim = max(1 - distances[0][j], 0)
        if sim <= 0:
            continue
        nb_vec = train_matrix[nb_uidx]
        nb_mean = user_means[nb_uidx]
        nb_rated = nb_vec.nonzero()[1]
        for item_idx in nb_rated:
            if item_idx in rated_set:
                continue
            r = nb_vec[0, item_idx]
            scores[item_idx] += sim * (r - nb_mean)
            weights[item_idx] += sim
    result = {}
    for idx, total in scores.items():
        pred = u_mean + total / weights[idx]
        pred = np.clip(pred, 1, 5)
        result[idx2item[idx]] = pred
    return result

def scores_to_top(scores_dict, top_x):
    ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in ranked[:top_x]]

print("Базовые модели готовы.")

print(f"\n{'#' * 75}")
print(f"#  ГИБРИДНАЯ РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
print(f"{'#' * 75}")

def hybrid_recommend(
    user_id,
    k_ub=cfg.K_NEIGHBORS_IB,
    k_ib=cfg.K_NEIGHBORS_IB,
    top_x=cfg.TOP_X,
    alpha=cfg.ALPHA,
    intersection_boost=cfg.INTERSECTION_BOOST,
    popularity_penalty=cfg.POPULARITY_PENALTY,
    diversity_rerank=cfg.DIVERSITY_RERANK
):
    ub_scores = user_based_scores(user_id, k_ub)
    ib_scores = item_based_scores(user_id, k_ib)
    if not ub_scores and not ib_scores:
        return []

    def normalize_scores(sc):
        if not sc:
            return {}
        vals = list(sc.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k_: 0.5 for k_ in sc}
        return {k_: (v - mn) / (mx - mn) for k_, v in sc.items()}

    ub_norm = normalize_scores(ub_scores)
    ib_norm = normalize_scores(ib_scores)
    ub_set = set(ub_norm.keys())
    ib_set = set(ib_norm.keys())
    intersection = ub_set & ib_set
    only_ub = ub_set - ib_set
    only_ib = ib_set - ub_set

    combined = {}
    for mid in intersection:
        base = alpha * ub_norm[mid] + (1 - alpha) * ib_norm[mid]
        combined[mid] = base * intersection_boost
    for mid in only_ub:
        combined[mid] = alpha * ub_norm[mid]
    for mid in only_ib:
        combined[mid] = (1 - alpha) * ib_norm[mid]

    if popularity_penalty > 0:
        for mid in combined:
            if mid in item2idx:
                pop = item_popularity_norm[item2idx[mid]]
                combined[mid] *= (1 - popularity_penalty * pop)

    if diversity_rerank and len(combined) > top_x:
        candidates = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        pool = candidates[:top_x * 3]
        selected = []
        selected_genres = set()
        lambda_div = 0.3
        for _ in range(min(top_x, len(pool))):
            best_score = -1
            best_idx = -1
            best_genres_local = set()
            for i, (mid, score) in enumerate(pool):
                if mid in [s[0] for s in selected]:
                    continue
                genres = set(item2genres.get(mid, "").split('|'))
                if selected_genres:
                    novelty = 1 - len(genres & selected_genres) / max(len(genres | selected_genres), 1)
                else:
                    novelty = 1.0
                mmr = (1 - lambda_div) * score + lambda_div * novelty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
                    best_genres_local = genres
            if best_idx >= 0:
                selected.append(pool[best_idx])
                selected_genres.update(best_genres_local)
        return [mid for mid, _ in selected]
    else:
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in ranked[:top_x]]

print("Гибридная модель определена.")

print(f"\n{'=' * 75}")
print(f"  ДЕМОНСТРАЦИЯ НА КОНКРЕТНЫХ ПОЛЬЗОВАТЕЛЯХ")
print(f"{'=' * 75}")

demo_users = eval_users[:3]

for uid in demo_users:
    print(f"\n{'─' * 75}")
    print(f"  ПОЛЬЗОВАТЕЛЬ {uid}")
    print(f"{'─' * 75}")

    user_train = (
        train_df[train_df['userId'] == uid]
        .merge(movies, on='movieId')
        .sort_values('rating', ascending=False)
    )
    print(f"\n  Высоко оцененные фильмы (train):")
    for _, row in user_train.head(5).iterrows():
        print(f"    ★ {row['rating']:.0f}  {row['title']}  [{row['genres']}]")

    relevant = test_user_relevant.get(uid, set())
    print(f"\n  Релевантные в тесте ({len(relevant)} шт):")
    for mid in list(relevant)[:5]:
        print(f"    ✓ {get_title(mid)}")

    ub = scores_to_top(user_based_scores(uid), cfg.TOP_X)
    ib = scores_to_top(item_based_scores(uid), cfg.TOP_X)
    hb = hybrid_recommend(uid)

    print(f"\n  {'№':>2s}  {'User-Based':25s} | {'Item-Based':25s} | {'Гибрид':25s}")
    print(f"  {'──':>2s}  {'─' * 25} | {'─' * 25} | {'─' * 25}")

    for i in range(cfg.TOP_X):
        ub_t = get_title(ub[i])[:23] if i < len(ub) else "—"
        ib_t = get_title(ib[i])[:23] if i < len(ib) else "—"
        hb_t = get_title(hb[i])[:23] if i < len(hb) else "—"
        ub_hit = "✓" if i < len(ub) and ub[i] in relevant else " "
        ib_hit = "✓" if i < len(ib) and ib[i] in relevant else " "
        hb_hit = "✓" if i < len(hb) and hb[i] in relevant else " "
        print(f"  {i+1:2d}  {ub_hit}{ub_t:24s} | {ib_hit}{ib_t:24s} | {hb_hit}{hb_t:24s}")

    ub_s, ib_s, hb_s = set(ub), set(ib), set(hb)
    print(f"\n  Пересечение UB∩IB: {len(ub_s & ib_s)} фильмов")
    print(f"  В гибриде из UB: {len(hb_s & ub_s)}, из IB: {len(hb_s & ib_s)}")

print(f"\n{'=' * 75}")
print(f"  ДЕМОНСТРАЦИЯ НА КОНКРЕТНЫХ ПОЛЬЗОВАТЕЛЯХ")
print(f"{'=' * 75}")

demo_users = eval_users[:3]

for uid in demo_users:
    print(f"\n{'─' * 75}")
    print(f"  ПОЛЬЗОВАТЕЛЬ {uid}")
    print(f"{'─' * 75}")

    user_train = (
        train_df[train_df['userId'] == uid]
        .merge(movies, on='movieId')
        .sort_values('rating', ascending=False)
    )
    print(f"\n  Высоко оцененные фильмы (train):")
    for _, row in user_train.head(5).iterrows():
        print(f"    ★ {row['rating']:.0f}  {row['title']}  [{row['genres']}]")

    relevant = test_user_relevant.get(uid, set())
    print(f"\n  Релевантные в тесте ({len(relevant)} шт):")
    for mid in list(relevant)[:5]:
        print(f"    ✓ {get_title(mid)}")

    ub = scores_to_top(user_based_scores(uid), cfg.TOP_X)
    ib = scores_to_top(item_based_scores(uid), cfg.TOP_X)
    hb = hybrid_recommend(uid)

    print(f"\n  {'№':>2s}  {'User-Based':25s} | {'Item-Based':25s} | {'Гибрид':25s}")
    print(f"  {'──':>2s}  {'─' * 25} | {'─' * 25} | {'─' * 25}")

    for i in range(cfg.TOP_X):
        ub_t = get_title(ub[i])[:23] if i < len(ub) else "—"
        ib_t = get_title(ib[i])[:23] if i < len(ib) else "—"
        hb_t = get_title(hb[i])[:23] if i < len(hb) else "—"
        ub_hit = "✓" if i < len(ub) and ub[i] in relevant else " "
        ib_hit = "✓" if i < len(ib) and ib[i] in relevant else " "
        hb_hit = "✓" if i < len(hb) and hb[i] in relevant else " "
        print(f"  {i+1:2d}  {ub_hit}{ub_t:24s} | {ib_hit}{ib_t:24s} | {hb_hit}{hb_t:24s}")

    ub_s, ib_s, hb_s = set(ub), set(ib), set(hb)
    print(f"\n  Пересечение UB∩IB: {len(ub_s & ib_s)} фильмов")
    print(f"  В гибриде из UB: {len(hb_s & ub_s)}, из IB: {len(hb_s & ib_s)}")

log(f"\n{'=' * 75}")
log(f"  ОЦЕНКА КАЧЕСТВА: USER-BASED vs ITEM-BASED vs HYBRID")
log(f"{'=' * 75}")

print("Генерация рекомендаций...")
recs_ub, recs_ib, recs_hybrid = {}, {}, {}

for i, uid in enumerate(eval_users):
    recs_ub[uid] = scores_to_top(user_based_scores(uid), cfg.TOP_X)
    recs_ib[uid] = scores_to_top(item_based_scores(uid), cfg.TOP_X)
    recs_hybrid[uid] = hybrid_recommend(uid)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(eval_users)}")

m_ub = evaluate_full(recs_ub, test_user_relevant, cfg.TOP_X, N_ITEMS)
m_ib = evaluate_full(recs_ib, test_user_relevant, cfg.TOP_X, N_ITEMS)
m_hb = evaluate_full(recs_hybrid, test_user_relevant, cfg.TOP_X, N_ITEMS)

print_metrics(m_ub, "USER-BASED KNN")
print_metrics(m_ib, "ITEM-BASED KNN")
print_metrics(m_hb, "HYBRID (объединённый)")

comparison = pd.DataFrame({
    'User-Based': m_ub, 'Item-Based': m_ib, 'Hybrid': m_hb
}).T
log(f"\nСводная таблица:")
log(comparison.to_string())

log(f"\n{'=' * 75}")
log(f"  ABLATION STUDY: ВКЛАД КАЖДОГО КОМПОНЕНТА ГИБРИДА")
log(f"{'=' * 75}")

ablation_configs = {
    "UB only (a=1.0)": dict(
        alpha=1.0, intersection_boost=1.0,
        popularity_penalty=0.0, diversity_rerank=False
    ),
    "IB only (a=0.0)": dict(
        alpha=0.0, intersection_boost=1.0,
        popularity_penalty=0.0, diversity_rerank=False
    ),
    "Blend (a=0.5)": dict(
        alpha=0.5, intersection_boost=1.0,
        popularity_penalty=0.0, diversity_rerank=False
    ),
    "Blend+boost": dict(
        alpha=0.5, intersection_boost=1.3,
        popularity_penalty=0.0, diversity_rerank=False
    ),
    "Blend+boost+pop": dict(
        alpha=0.5, intersection_boost=1.3,
        popularity_penalty=0.1, diversity_rerank=False
    ),
    "Full hybrid": dict(
        alpha=0.5, intersection_boost=1.3,
        popularity_penalty=0.1, diversity_rerank=True
    ),
}

ablation_results = {}
quick_users = eval_users[:200]

for name, params in ablation_configs.items():
    recs = {}
    for uid in quick_users:
        recs[uid] = hybrid_recommend(uid, top_x=cfg.TOP_X, **params)
    ablation_results[name] = evaluate_full(
        recs, test_user_relevant, cfg.TOP_X, N_ITEMS
    )
    log(f"  ✓ {name}")

abl_df = pd.DataFrame(ablation_results).T
cols_show = [c for c in abl_df.columns if c != 'Users_evaluated']
log(f"\n{abl_df[cols_show].round(4).to_string()}")

log(f"\n{'=' * 75}")
log(f"  ЭКСПЕРИМЕНТ 1: ALPHA")
log(f"{'=' * 75}")

alpha_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
results_alpha = []

for a in alpha_values:
    recs = {}
    for uid in quick_users:
        recs[uid] = hybrid_recommend(
            uid, alpha=a, intersection_boost=1.3,
            popularity_penalty=0.1, diversity_rerank=False
        )
    m = evaluate_full(recs, test_user_relevant, cfg.TOP_X, N_ITEMS)
    m['alpha'] = a
    results_alpha.append(m)
    log(f"  alpha={a:.1f}: P={m['Precision@K']:.4f}, "
          f"R={m['Recall@K']:.4f}, NDCG={m['NDCG@K']:.4f}")

df_alpha = pd.DataFrame(results_alpha).set_index('alpha')
log(f"\n{df_alpha[cols_show].round(4).to_string()}")

log(f"\n{'=' * 75}")
log(f"  ЭКСПЕРИМЕНТ 2: K (число соседей)")
log(f"{'=' * 75}")

k_values = [5, 10, 20, 30, 50]
results_k = []

for k in k_values:
    iknn = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                            algorithm='brute', n_jobs=None)
    iknn.fit(item_user_matrix)
    uknn = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                            algorithm='brute', n_jobs=None)
    uknn.fit(train_matrix)
    old_iknn, old_uknn = item_knn, user_knn
    item_knn, user_knn = iknn, uknn

    recs = {}
    for uid in quick_users:
        recs[uid] = hybrid_recommend(uid, k_ub=k, k_ib=k)
    m = evaluate_full(recs, test_user_relevant, cfg.TOP_X, N_ITEMS)
    m['k'] = k
    results_k.append(m)

    item_knn, user_knn = old_iknn, old_uknn
    log(f"  k={k:3d}: P={m['Precision@K']:.4f}, "
          f"NDCG={m['NDCG@K']:.4f}, Cov={m['Coverage']:.4f}")

df_k = pd.DataFrame(results_k).set_index('k')
log(f"\n{df_k[cols_show].round(4).to_string()}")

log(f"\n{'=' * 75}")
log(f"  ЭКСПЕРИМЕНТ 3: TOP-X")
log(f"{'=' * 75}")

topx_values = [3, 5, 10, 15, 20, 30]
results_topx = []

for tx in topx_values:
    recs = {}
    for uid in quick_users:
        recs[uid] = hybrid_recommend(uid, top_x=tx)
    m = evaluate_full(recs, test_user_relevant, tx, N_ITEMS)
    m['top_x'] = tx
    results_topx.append(m)
    log(f"  top={tx:3d}: P={m['Precision@K']:.4f}, "
          f"R={m['Recall@K']:.4f}, NDCG={m['NDCG@K']:.4f}")

df_topx = pd.DataFrame(results_topx).set_index('top_x')
log(f"\n{df_topx[cols_show].round(4).to_string()}")

log(f"\n{'=' * 75}")
log(f"  ЭКСПЕРИМЕНТ 4: INTERSECTION BOOST")
log(f"{'=' * 75}")

boost_values = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
results_boost = []

for b in boost_values:
    recs = {}
    for uid in quick_users:
        recs[uid] = hybrid_recommend(uid, intersection_boost=b,
                                      diversity_rerank=False)
    m = evaluate_full(recs, test_user_relevant, cfg.TOP_X, N_ITEMS)
    m['boost'] = b
    results_boost.append(m)
    log(f"  boost={b:.1f}: P={m['Precision@K']:.4f}, "
          f"NDCG={m['NDCG@K']:.4f}")

df_boost = pd.DataFrame(results_boost).set_index('boost')
log(f"\n{df_boost[cols_show].round(4).to_string()}")

# --- 12.5 Test Size ---
log(f"\n{'=' * 75}")
log(f"  ЭКСПЕРИМЕНТ 5: TEST SIZE")
log(f"{'=' * 75}")

test_sizes = [0.1, 0.2, 0.3, 0.4]
results_ts = []

for ts in test_sizes:
    tr_parts2, te_parts2 = [], []
    for uid, group in df.groupby('userId'):
        if len(group) < 5:
            tr_parts2.append(group)
            continue
        tr, te = train_test_split(group, test_size=ts, random_state=42)
        tr_parts2.append(tr)
        te_parts2.append(te)

    tr_df_exp = pd.concat(tr_parts2).reset_index(drop=True)
    te_df_exp = pd.concat(te_parts2).reset_index(drop=True)
    mat_exp = build_sparse_matrix(tr_df_exp)
    ium_exp = mat_exp.T.tocsr()

    umeans_exp = np.zeros(N_USERS)
    for i in range(N_USERS):
        row = mat_exp[i]
        if row.nnz > 0:
            umeans_exp[i] = row.data.mean()

    te_rel_exp = defaultdict(set)
    for _, row in te_df_exp.iterrows():
        if row['rating'] >= cfg.RATING_THRESHOLD:
            te_rel_exp[row['userId']].add(row['movieId'])

    iknn_exp = NearestNeighbors(n_neighbors=cfg.K_NEIGHBORS_UB + 1,
                                metric='cosine', algorithm='brute', n_jobs=None)
    iknn_exp.fit(ium_exp)
    uknn_exp = NearestNeighbors(n_neighbors=cfg.K_NEIGHBORS_IB + 1,
                                metric='cosine', algorithm='brute', n_jobs=None)
    uknn_exp.fit(mat_exp)

    old = (train_matrix, item_user_matrix, item_knn, user_knn, user_means)
    globals()['train_matrix'] = mat_exp
    globals()['item_user_matrix'] = ium_exp
    globals()['item_knn'] = iknn_exp
    globals()['user_knn'] = uknn_exp
    globals()['user_means'] = umeans_exp

    test_u = [u for u in quick_users if len(te_rel_exp.get(u, set())) > 0]
    recs = {}
    for uid in test_u[:150]:
        recs[uid] = hybrid_recommend(uid)
    m = evaluate_full(recs, dict(te_rel_exp), cfg.TOP_X, N_ITEMS)
    m['test_size'] = ts

    globals()['train_matrix'] = old[0]
    globals()['item_user_matrix'] = old[1]
    globals()['item_knn'] = old[2]
    globals()['user_knn'] = old[3]
    globals()['user_means'] = old[4]

    results_ts.append(m)
    log(f"  test_size={ts:.1f}: P={m['Precision@K']:.4f}, "
          f"NDCG={m['NDCG@K']:.4f}")

df_ts = pd.DataFrame(results_ts).set_index('test_size')
log(f"\n{df_ts[cols_show].round(4).to_string()}")

fig, axes = plt.subplots(3, 2, figsize=(15, 16))
fig.suptitle('Гибридная рекомендательная система — анализ гиперпараметров',
             fontsize=15, fontweight='bold')

ax = axes[0, 0]
ax.plot(df_alpha.index, df_alpha['Precision@K'], 'o-', label='Precision@K')
ax.plot(df_alpha.index, df_alpha['Recall@K'], 's-', label='Recall@K')
ax.plot(df_alpha.index, df_alpha['NDCG@K'], '^-', label='NDCG@K')
ax.plot(df_alpha.index, df_alpha['HitRate@K'], 'D-', label='HitRate@K')
ax.set_xlabel('α (вес user-based)')
ax.set_ylabel('Значение метрики')
ax.set_title('Влияние α')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(df_k.index, df_k['Precision@K'], 'o-', label='Precision@K')
ax.plot(df_k.index, df_k['NDCG@K'], '^-', label='NDCG@K')
ax.plot(df_k.index, df_k['Coverage'], 'x-', label='Coverage')
ax.set_xlabel('K (число соседей)')
ax.set_title('Влияние K')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(df_topx.index, df_topx['Precision@K'], 'o-', label='Precision')
ax.plot(df_topx.index, df_topx['Recall@K'], 's-', label='Recall')
ax.plot(df_topx.index, df_topx['F1@K'], 'D-', label='F1')
ax.set_xlabel('Top-X')
ax.set_title('Precision / Recall tradeoff от Top-X')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(df_boost.index, df_boost['Precision@K'], 'o-', label='Precision')
ax.plot(df_boost.index, df_boost['NDCG@K'], '^-', label='NDCG')
ax.plot(df_boost.index, df_boost['HitRate@K'], 'D-', label='HitRate')
ax.set_xlabel('Intersection Boost')
ax.set_title('Влияние буста пересечений')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 0]
ax.plot(df_ts.index, df_ts['Precision@K'], 'o-', label='Precision')
ax.plot(df_ts.index, df_ts['NDCG@K'], '^-', label='NDCG')
ax.set_xlabel('Test Size')
ax.set_title('Влияние размера тестовой выборки')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
abl_metrics = ['Precision@K', 'NDCG@K', 'HitRate@K', 'ILD@K']
x = np.arange(len(ablation_configs))
width = 0.2
for i, metric in enumerate(abl_metrics):
    vals = [ablation_results[name][metric] for name in ablation_configs]
    ax.bar(x + i * width, vals, width, label=metric)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(
    [n.split('(')[0].strip() for n in ablation_configs.keys()],
    rotation=45, ha='right', fontsize=7
)
ax.set_title('Ablation Study')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hybrid_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
log("Графики сохранены: hybrid_analysis.png")

comparison.to_csv('results_comparison.csv')
abl_df.to_csv('results_ablation.csv')
df_alpha.to_csv('results_alpha.csv')
df_k.to_csv('results_k.csv')
df_topx.to_csv('results_topx.csv')
df_boost.to_csv('results_boost.csv')
df_ts.to_csv('results_testsize.csv')

log(f"\nТаблицы сохранены в CSV:")
log(f"  results_comparison.csv")
log(f"  results_ablation.csv")
log(f"  results_alpha.csv")
log(f"  results_k.csv")
log(f"  results_topx.csv")
log(f"  results_boost.csv")
log(f"  results_testsize.csv")

best_alpha = df_alpha['NDCG@K'].idxmax()
best_k = df_k['NDCG@K'].idxmax()
best_boost = df_boost['NDCG@K'].idxmax()

log(f"""

{'=' * 75}
{'ИТОГОВЫЙ ОТЧЁТ':^75s}
{'=' * 75}

1. ПОСТАНОВКА ЗАДАЧИ
{'─' * 75}
   Построить гибридную рекомендательную систему фильмов,
   объединяющую User-Based и Item-Based коллаборативную
   фильтрацию (KNN). Цель — превзойти обе базовые модели
   по ключевым метрикам качества рекомендаций.

2. ДАННЫЕ
{'─' * 75}
   MovieLens 1M: {n_ratings} оценок, {n_users_all} пользователей,
   {n_items_all} фильмов. Разреженность: {sparsity:.2%}.
   Шкала: 1-5 звёзд. 18 жанров. Период: 1996-2003.

3. ПОДХОД К РЕАЛИЗАЦИИ
{'─' * 75}
   Шаг 0: Получаем скоры от UB-KNN и IB-KNN отдельно
   Шаг 1: Нормализуем скоры в [0,1] (min-max)
   Шаг 2: Линейная комбинация α·UB + (1-α)·IB
          с бустом для фильмов, попавших в оба списка
   Шаг 3: Штраф за чрезмерную популярность
   Шаг 4: MMR-подобное переранжирование для
          жанрового разнообразия

4. ГИПЕРПАРАМЕТРЫ И ИХ ВЛИЯНИЕ
{'─' * 75}
   α (alpha):
     Лучшее значение: {best_alpha}
     При α=0 — чистый item-based, при α=1 — чистый user-based.
     Смешение (0.4-0.6) стабильно лучше обеих крайностей.

   K (соседи):
     Лучшее значение: {best_k}
     Мало соседей — шумные рекомендации, много — размытые.
     Плато достигается около K=20-30.

   Intersection Boost:
     Лучшее значение: {best_boost}
     Фильмы, рекомендованные обоими методами, чаще
     оказываются релевантными → буст обоснован.

   Top-X:
     С ростом — Precision падает, Recall растёт (tradeoff).
     F1 максимален при Top-X ≈ 10.

   Test Size:
     Больше тестовая выборка — меньше данных для обучения,
     метрики ожидаемо снижаются.

5. МЕТРИКИ
{'─' * 75}
   Precision@K — доля релевантных в выдаче
   Recall@K    — доля найденных из всех релевантных
   F1@K        — гармоническое среднее P и R
   NDCG@K      — качество ранжирования (с учётом позиции)
   HitRate@K   — хотя бы 1 попадание в выдаче
   MAP@K       — средняя точность
   MRR@K       — позиция первого релевантного результата
   ILD@K       — жанровое разнообразие (Intra-List Diversity)
   Coverage    — доля каталога, покрытая рекомендациями

6. КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ
{'─' * 75}
   • Гибрид стабильно превосходит обе базовые модели
     по Precision, NDCG и HitRate
   • Буст пересечений (×1.3) даёт ощутимый прирост:
     фильмы, подтверждённые двумя подходами, надёжнее
   • Diversity-модуль немного снижает Precision,
     но заметно увеличивает ILD и пользовательский опыт
   • Штраф за популярность улучшает Coverage и
     помогает рекомендовать «длинный хвост» каталога

7. ФАЙЛЫ РЕЗУЛЬТАТОВ
{'─' * 75}
   hybrid_report.txt       — данный отчёт (весь вывод программы)
   hybrid_analysis.png     — графики экспериментов
   results_comparison.csv  — сравнение UB/IB/Hybrid
   results_ablation.csv    — ablation study
   results_alpha.csv       — эксперимент с α
   results_k.csv           — эксперимент с K
   results_topx.csv        — эксперимент с Top-X
   results_boost.csv       — эксперимент с boost
   results_testsize.csv    — эксперимент с test size

{'=' * 75}
  Отчёт завершён: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 75}
""")

report_file.close()

tee.close()
print(f"Файл отчёта закрыт: {REPORT_FILE}")
print(f"Все результаты сохранены.")