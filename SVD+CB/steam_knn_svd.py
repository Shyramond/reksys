import os
import re
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, normalize

DATA_PATH = "steam_top_games_2026.csv"
REPORT_FILE = "steam_knn_svd_report.txt"
IMAGE_CACHE_DIR = ".image_cache"
IMAGE_HIST_BINS = 8
IMAGE_FEATURE_DIM = IMAGE_HIST_BINS * 3
TOP_K = 10
TEST_QUERIES = 200
SVD_COMPONENTS = [20, 50, 100]
MAX_TEXT_FEATURES = 7000
RANDOM_STATE = 42

report_file = open(REPORT_FILE, "w", encoding="utf-8")

def log(message="", end="\n"):
    text = str(message)
    report_file.write(text + end)
    report_file.flush()
    print(text, end=end)


def clean_text(value: str) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def remove_token(text: str, token: str) -> str:
    if not text or not token:
        return text
    pattern = re.compile(rf"\b{re.escape(token.lower())}\b")
    return pattern.sub("", text.lower())


def build_item_text(row: pd.Series, hidden_tag: str | None = None) -> str:
    name = clean_text(row.get("name", ""))
    short_description = clean_text(row.get("short_description", ""))
    genres = clean_text(row.get("genres", ""))
    categories = clean_text(row.get("categories", ""))
    tags = clean_text(row.get("tags", ""))
    developer = clean_text(row.get("developer", ""))
    publisher = clean_text(row.get("publisher", ""))
    owners = clean_text(row.get("estimated_owners", ""))
    if hidden_tag:
        name = remove_token(name, hidden_tag)
        short_description = remove_token(short_description, hidden_tag)
        genres = remove_token(genres, hidden_tag)
        categories = remove_token(categories, hidden_tag)
        tags = remove_token(tags, hidden_tag)
        owners = remove_token(owners, hidden_tag)
    platform_tokens = []
    for platform in ("platforms_win", "platforms_mac", "platforms_linux"):
        if row.get(platform, False):
            platform_tokens.append(platform.replace("platforms_", ""))
    free_flag = "free" if row.get("is_free", False) else "paid"
    coming = "coming_soon" if row.get("coming_soon", False) else "released"
    parts = [name, genres, categories, tags, developer, publisher, owners, short_description, free_flag, coming] + platform_tokens
    return " ".join([p for p in parts if p])


def image_to_histogram(image: Image.Image, bins: int = IMAGE_HIST_BINS) -> np.ndarray:
    image = image.convert("RGB").resize((128, 128))
    arr = np.asarray(image)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0, 255))
        histograms.append(hist.astype(np.float32))
    hist = np.concatenate(histograms)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def download_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=15) as response:
        data = response.read()
    return Image.open(BytesIO(data))


def build_image_features(df: pd.DataFrame) -> np.ndarray:
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    image_features = np.zeros((len(df), IMAGE_FEATURE_DIM), dtype=np.float32)
    for idx, url in enumerate(df["header_image"].fillna("")):
        if not url or not isinstance(url, str):
            continue
        cache_name = f"image_{idx}_{int(df.iloc[idx].get('app_id', idx))}"
        cache_file = os.path.join(IMAGE_CACHE_DIR, f"{cache_name}.npy")
        if os.path.exists(cache_file):
            try:
                image_features[idx] = np.load(cache_file)
                continue
            except Exception:
                pass
        try:
            img = download_image(url)
            hist = image_to_histogram(img)
            np.save(cache_file, hist)
            image_features[idx] = hist
        except Exception as exc:
            log(f"Warning: image load failed for idx={idx}, url={url}: {exc}")
    return image_features


def build_feature_matrix(df: pd.DataFrame):
    df = df.copy()
    df["genres"] = df["genres"].fillna("")
    df["categories"] = df["categories"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["developer"] = df["developer"].fillna("")
    df["publisher"] = df["publisher"].fillna("")
    df["estimated_owners"] = df["estimated_owners"].fillna("")

    df["feature_text"] = df.apply(lambda row: build_item_text(row), axis=1)
    vectorizer = TfidfVectorizer(max_features=MAX_TEXT_FEATURES, token_pattern=r"(?u)\b[\w\-]+\b")
    X_text = vectorizer.fit_transform(df["feature_text"])

    numeric_columns = [
        "price_usd",
        "discount_pct",
        "metacritic_score",
        "recommendations",
        "positive_reviews",
        "negative_reviews",
        "avg_playtime_forever",
        "avg_playtime_2weeks",
        "median_playtime",
        "peak_ccu",
        "required_age",
        "dlc_count",
        "achievements",
    ]
    numeric = df[numeric_columns].fillna(0).astype(float).copy()
    numeric["recommendations"] = np.log1p(numeric["recommendations"])
    numeric["positive_reviews"] = np.log1p(numeric["positive_reviews"])
    numeric["negative_reviews"] = np.log1p(numeric["negative_reviews"])
    numeric["avg_playtime_forever"] = np.log1p(numeric["avg_playtime_forever"])
    numeric["avg_playtime_2weeks"] = np.log1p(numeric["avg_playtime_2weeks"])
    numeric["median_playtime"] = np.log1p(numeric["median_playtime"])
    numeric["peak_ccu"] = np.log1p(numeric["peak_ccu"])

    scalar = MinMaxScaler()
    X_num = sparse.csr_matrix(scalar.fit_transform(numeric))

    bool_columns = ["platforms_win", "platforms_mac", "platforms_linux", "is_free", "coming_soon"]
    X_bool = sparse.csr_matrix(df[bool_columns].astype(float).values)

    owner_dummies = pd.get_dummies(df["estimated_owners"], prefix="owners")
    X_owner = sparse.csr_matrix(owner_dummies.values)

    X_extra = sparse.hstack([X_num, X_bool, X_owner], format="csr")
    image_features = build_image_features(df)
    X_base = normalize(sparse.hstack([X_text, X_extra], format="csr"), norm="l2", axis=1)
    X_content = normalize(sparse.hstack([X_text, sparse.csr_matrix(image_features), X_extra], format="csr"), norm="l2", axis=1)
    return df, vectorizer, X_text, image_features, X_extra, X_base, X_content, numeric, bool_columns, owner_dummies.columns.tolist()


def build_hidden_tag_queries(df: pd.DataFrame, candidate_ids: list):
    query_rows = []
    relevant_sets = []
    query_tags = []
    for idx in candidate_ids:
        row = df.iloc[idx]
        tags = [tag.strip().lower() for tag in str(row["tags"]).split(",") if tag.strip()]
        tags = [tag for tag in tags if tag]
        if len(tags) < 2:
            continue
        hidden_tag = tags[0]
        query_rows.append((idx, hidden_tag, build_item_text(row, hidden_tag=hidden_tag)))

    for idx, hidden_tag, _ in query_rows:
        relevant = set(df.index[df["tags"].str.lower().str.contains(fr"\b{re.escape(hidden_tag)}\b", na=False)])
        relevant.discard(idx)
        relevant_sets.append(relevant)
        query_tags.append(hidden_tag)

    return query_rows, relevant_sets, query_tags


def build_full_query_matrix(query_text_matrix, df: pd.DataFrame, query_rows, numeric, bool_columns, owner_columns):
    query_numeric = numeric.iloc[[idx for idx, _, _ in query_rows]].values
    query_bool = df[bool_columns].astype(float).iloc[[idx for idx, _, _ in query_rows]].values
    query_owner = pd.get_dummies(df["estimated_owners"].iloc[[idx for idx, _, _ in query_rows]]).reindex(columns=owner_columns, fill_value=0).values

    query_matrix = sparse.hstack([
        query_text_matrix,
        sparse.csr_matrix(query_numeric),
        sparse.csr_matrix(query_bool),
        sparse.csr_matrix(query_owner)
    ], format="csr")
    return normalize(query_matrix, norm="l2", axis=1)


def build_content_query_matrix(query_text_matrix, df: pd.DataFrame, query_rows, numeric, bool_columns, owner_columns, image_features):
    query_img = image_features[[idx for idx, _, _ in query_rows]]
    query_numeric = numeric.iloc[[idx for idx, _, _ in query_rows]].values
    query_bool = df[bool_columns].astype(float).iloc[[idx for idx, _, _ in query_rows]].values
    query_owner = pd.get_dummies(df["estimated_owners"].iloc[[idx for idx, _, _ in query_rows]]).reindex(columns=owner_columns, fill_value=0).values

    query_matrix = sparse.hstack([
        query_text_matrix,
        sparse.csr_matrix(query_img),
        sparse.csr_matrix(query_numeric),
        sparse.csr_matrix(query_bool),
        sparse.csr_matrix(query_owner)
    ], format="csr")
    return normalize(query_matrix, norm="l2", axis=1)


def build_svd_index(X_text, n_components, n_neighbors=TOP_K * 5 + 1):
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_text_reduced = normalize(svd.fit_transform(X_text), norm="l2", axis=1)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(X_text_reduced)
    return svd, X_text_reduced, knn


def get_svd_candidates(seed_idx, knn, X_reduced, top_n=50):
    distances, indices = knn.kneighbors(X_reduced[seed_idx].reshape(1, -1), n_neighbors=top_n + 1)
    return [i for i in indices[0] if i != seed_idx][:top_n]


def refine_candidates_by_similarity(seed_idx, candidates, full_matrix, top_n=20):
    if sparse.issparse(full_matrix):
        seed_vec = full_matrix[seed_idx]
        cand_mat = full_matrix[candidates]
        sims = cand_mat.dot(seed_vec.T).toarray().flatten()
    else:
        seed_vec = full_matrix[seed_idx]
        sims = np.dot(full_matrix[candidates], seed_vec)
    ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:top_n]]


def tag_overlap(df: pd.DataFrame, seed_idx: int, cand_idx: int) -> float:
    seed_tags = {tag.strip().lower() for tag in str(df.loc[seed_idx, "tags"]).split(",") if tag.strip()}
    cand_tags = {tag.strip().lower() for tag in str(df.loc[cand_idx, "tags"]).split(",") if tag.strip()}
    if not seed_tags and not cand_tags:
        return 0.0
    return len(seed_tags & cand_tags) / max(1, len(seed_tags | cand_tags))


def personalize_candidates(seed_idx, candidates, X_img, df, top_k=TOP_K):
    seed_img = X_img[seed_idx]
    if np.linalg.norm(seed_img) < 1e-8:
        return candidates[:top_k]
    cand_imgs = X_img[candidates]
    image_sims = cand_imgs.dot(seed_img)
    score_pairs = []
    for i, cand in enumerate(candidates):
        tag_score = tag_overlap(df, seed_idx, cand)
        score_pairs.append((cand, image_sims[i], tag_score))
    ordered = sorted(score_pairs, key=lambda x: (x[1], x[2]), reverse=True)
    return [cand for cand, _, _ in ordered][:top_k]


def sequential_svd_then_refine_then_content(seed_idx, svd_knn, X_reduced, full_matrix, X_img, df, top_k=TOP_K):
    candidates = get_svd_candidates(seed_idx, svd_knn, X_reduced, top_n=50)
    refined = refine_candidates_by_similarity(seed_idx, candidates, full_matrix, top_n=30)
    personalized = personalize_candidates(seed_idx, refined, X_img, df, top_k=top_k)
    return personalized


def sequential_content_then_refine_then_svd(seed_idx, content_knn, X_content, svd_knn, X_reduced, full_matrix, X_img, df, top_k=TOP_K):
    distances, indices = content_knn.kneighbors(X_content[seed_idx], n_neighbors=50 + 1)
    candidates = [i for i in indices[0] if i != seed_idx][:50]
    refined = refine_candidates_by_similarity(seed_idx, candidates, full_matrix, top_n=30)
    personalized = personalize_candidates(seed_idx, refined, X_img, df, top_k=top_k)
    return personalized


def evaluate_recommender(recommender, query_ids, relevant_sets, top_k=TOP_K):
    metrics = defaultdict(list)
    for query_pos, (seed_idx, _, _) in enumerate(query_ids):
        recs = recommender(seed_idx)
        relevant = relevant_sets[query_pos]
        if not relevant:
            continue
        metrics["Precision@K"].append(precision_at_k(recs, relevant, top_k))
        metrics["Recall@K"].append(recall_at_k(recs, relevant, top_k))
        metrics["NDCG@K"].append(ndcg_at_k(recs, relevant, top_k))
        metrics["MRR@K"].append(mrr_at_k(recs, relevant, top_k))
    return {name: np.mean(vals) if vals else 0.0 for name, vals in metrics.items()}


def evaluate_neighbors(knn: NearestNeighbors, query_matrix, query_ids, relevant_sets, top_k=TOP_K):
    distances, indices = knn.kneighbors(query_matrix, n_neighbors=top_k + 1)
    metrics = defaultdict(list)
    for query_pos, (query_idx, hidden_tag, _) in enumerate(query_ids):
        candidates = [idx for idx in indices[query_pos] if idx != query_idx][:top_k]
        relevant = relevant_sets[query_pos]
        if not relevant:
            continue
        recs = candidates
        metrics["Precision@K"].append(precision_at_k(recs, relevant, top_k))
        metrics["Recall@K"].append(recall_at_k(recs, relevant, top_k))
        metrics["NDCG@K"].append(ndcg_at_k(recs, relevant, top_k))
        metrics["MRR@K"].append(mrr_at_k(recs, relevant, top_k))
    return {name: np.mean(vals) if vals else 0.0 for name, vals in metrics.items()}


def precision_at_k(recs, relevant, k):
    if not recs:
        return 0.0
    return len(set(recs[:k]) & relevant) / len(recs[:k])


def recall_at_k(recs, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recs[:k]) & relevant) / len(relevant)


def ndcg_at_k(recs, relevant, k):
    dcg = 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mrr_at_k(recs, relevant, k):
    for rank, item in enumerate(recs[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def print_results(name: str, metrics: dict):
    log(f"\n{name}")
    log("-" * len(name))
    for key, value in metrics.items():
        log(f"{key:10s}: {value:.4f}")


def main():
    df = pd.read_csv(DATA_PATH)
    log(f"Loaded {len(df)} Steam games from {DATA_PATH}")

    df, vectorizer, X_text, X_img, X_extra, X_base, X_content, numeric, bool_columns, owner_columns = build_feature_matrix(df)
    log(f"Base feature matrix: {X_base.shape[0]} items x {X_base.shape[1]} dimensions")
    log(f"Content feature matrix: {X_content.shape[0]} items x {X_content.shape[1]} dimensions")

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
    if len(candidate_indices) < TEST_QUERIES:
        raise ValueError("Not enough suitable query games for evaluation")

    np.random.seed(RANDOM_STATE)
    sampled_indices = np.random.choice(candidate_indices, size=TEST_QUERIES, replace=False)
    query_rows, relevant_sets, query_tags = build_hidden_tag_queries(df, sampled_indices)
    query_texts = [build_item_text(df.iloc[idx], hidden_tag=hidden_tag) for idx, hidden_tag, _ in query_rows]
    query_text_matrix = vectorizer.transform(query_texts)
    query_base = build_full_query_matrix(query_text_matrix, df, query_rows, numeric, bool_columns, owner_columns)
    query_content = build_content_query_matrix(query_text_matrix, df, query_rows, numeric, bool_columns, owner_columns, X_img)

    raw_knn = NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    raw_knn.fit(X_base)
    raw_metrics = evaluate_neighbors(raw_knn, query_base, query_rows, relevant_sets, TOP_K)
    print_results("Raw KNN on item metadata", raw_metrics)

    content_knn = NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    content_knn.fit(X_content)
    content_metrics = evaluate_neighbors(content_knn, query_content, query_rows, relevant_sets, TOP_K)
    print_results("Content-based KNN with image features", content_metrics)

    results = {"raw": raw_metrics, "content": content_metrics}
    svd_results = {}
    for n_components in SVD_COMPONENTS:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        X_text_reduced = svd.fit_transform(X_text)
        X_reduced = np.hstack([X_text_reduced, X_extra.toarray()])
        X_reduced = normalize(X_reduced, norm="l2", axis=1)

        query_text_reduced = svd.transform(query_text_matrix)
        query_extra = np.hstack([
            numeric.iloc[[idx for idx, _, _ in query_rows]].values,
            df[bool_columns].astype(float).iloc[[idx for idx, _, _ in query_rows]].values,
            pd.get_dummies(df["estimated_owners"].iloc[[idx for idx, _, _ in query_rows]]).reindex(columns=owner_columns, fill_value=0).values
        ])
        query_reduced = np.hstack([query_text_reduced, query_extra])
        query_reduced = normalize(query_reduced, norm="l2", axis=1)

        svd_knn = NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        svd_knn.fit(X_reduced)
        svd_metrics = evaluate_neighbors(svd_knn, query_reduced, query_rows, relevant_sets, TOP_K)
        print_results(f"SVD KNN with {n_components} components", svd_metrics)
        results[f"svd_{n_components}"] = svd_metrics
        svd_results[n_components] = svd_metrics

    best_svd_k = max(svd_results, key=lambda k: svd_results[k]["NDCG@K"])
    log(f"\nBest SVD dimension by NDCG@K: {best_svd_k}")
    best_svd = TruncatedSVD(n_components=best_svd_k, random_state=RANDOM_STATE)
    X_text_best = best_svd.fit_transform(X_text)
    X_text_best_norm = normalize(X_text_best, norm="l2", axis=1)
    query_text_best = best_svd.transform(query_text_matrix)

    # build final hybrid pipelines using SVD + content-based refinement
    svd_knn_best = NearestNeighbors(n_neighbors=TOP_K * 10 + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    svd_knn_best.fit(X_text_best_norm)

    def hybrid_svd_first(seed_idx):
        return sequential_svd_then_refine_then_content(seed_idx, svd_knn_best, X_text_best_norm, X_content, X_img, df, top_k=TOP_K)

    def hybrid_content_first(seed_idx):
        return sequential_content_then_refine_then_svd(seed_idx, content_knn, X_content, svd_knn_best, X_text_best_norm, X_content, X_img, df, top_k=TOP_K)

    hybrid_svd_metrics = evaluate_recommender(hybrid_svd_first, query_rows, relevant_sets, TOP_K)
    print_results("Hybrid SVD-first -> full refinement -> content personalization", hybrid_svd_metrics)
    results["hybrid_svd_first"] = hybrid_svd_metrics

    hybrid_content_metrics = evaluate_recommender(hybrid_content_first, query_rows, relevant_sets, TOP_K)
    print_results("Hybrid content-first -> full refinement -> personalization", hybrid_content_metrics)
    results["hybrid_content_first"] = hybrid_content_metrics

    X_best_combined = normalize(np.hstack([X_text_best, X_img, X_extra.toarray()]), axis=1)

    query_img = X_img[[idx for idx, _, _ in query_rows]]
    query_extra = np.hstack([
        numeric.iloc[[idx for idx, _, _ in query_rows]].values,
        df[bool_columns].astype(float).iloc[[idx for idx, _, _ in query_rows]].values,
        pd.get_dummies(df["estimated_owners"].iloc[[idx for idx, _, _ in query_rows]]).reindex(columns=owner_columns, fill_value=0).values
    ])
    query_best_combined = normalize(np.hstack([query_text_best, query_img, query_extra]), axis=1)

    combined_knn = NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    combined_knn.fit(X_best_combined)
    combined_metrics = evaluate_neighbors(combined_knn, query_best_combined, query_rows, relevant_sets, TOP_K)
    print_results("Combined SVD + image content KNN", combined_metrics)
    results["combined"] = combined_metrics

    summary = pd.DataFrame(results).T
    summary.index.name = "model"
    summary.to_csv("steam_knn_svd_comparison.csv")
    log(f"\nSaved comparison results to steam_knn_svd_comparison.csv")

    log("\nExample seed games and hidden tags:")
    for i, (idx, hidden_tag, _) in enumerate(query_rows[:5], start=1):
        row = df.iloc[idx]
        log(f"{i}. {row['name']} (hidden tag: {hidden_tag})")

    log("\nRecommendation examples for the first query:")
    example_idx, example_tag, _ = query_rows[0]
    query_desc = df.iloc[example_idx]["name"]
    log(f"Seed: {query_desc}, hidden tag: {example_tag}")

    raw_neighbors = raw_knn.kneighbors(query_base[0], n_neighbors=TOP_K + 1)[1][0]
    raw_neighbors = [n for n in raw_neighbors if n != example_idx][:TOP_K]
    log("Raw neighbors:")
    for n in raw_neighbors:
        log(f"  - {df.iloc[n]['name']} | tags={df.iloc[n]['tags']}")

    content_neighbors = content_knn.kneighbors(query_content[0], n_neighbors=TOP_K + 1)[1][0]
    content_neighbors = [n for n in content_neighbors if n != example_idx][:TOP_K]
    log("\nContent-based neighbors:")
    for n in content_neighbors:
        log(f"  - {df.iloc[n]['name']} | tags={df.iloc[n]['tags']}")

    combined_neighbors = combined_knn.kneighbors(query_best_combined[0:1], n_neighbors=TOP_K + 1)[1][0]
    combined_neighbors = [n for n in combined_neighbors if n != example_idx][:TOP_K]
    log("\nCombined SVD+image-content neighbors:")
    for n in combined_neighbors:
        log(f"  - {df.iloc[n]['name']} | tags={df.iloc[n]['tags']}")

    for n_components in SVD_COMPONENTS:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        X_text_reduced = svd.fit_transform(X_text)
        X_reduced = np.hstack([X_text_reduced, X_extra.toarray()])
        X_reduced = normalize(X_reduced, norm="l2", axis=1)
        query_text_reduced = svd.transform(query_text_matrix)
        query_extra = np.hstack([
            numeric.iloc[[idx for idx, _, _ in query_rows]].values,
            df[bool_columns].astype(float).iloc[[idx for idx, _, _ in query_rows]].values,
            pd.get_dummies(df["estimated_owners"].iloc[[idx for idx, _, _ in query_rows]]).reindex(columns=owner_columns, fill_value=0).values
        ])
        query_reduced = np.hstack([query_text_reduced, query_extra])
        query_reduced = normalize(query_reduced, norm="l2", axis=1)
        svd_knn = NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        svd_knn.fit(X_reduced)
        svd_neighbors = svd_knn.kneighbors(query_reduced[0:1], n_neighbors=TOP_K + 1)[1][0]
        svd_neighbors = [n for n in svd_neighbors if n != example_idx][:TOP_K]
        log(f"\nSVD ({n_components}) neighbors:")
        for n in svd_neighbors:
            log(f"  - {df.iloc[n]['name']} | tags={df.iloc[n]['tags']}")


if __name__ == "__main__":
    try:
        main()
    finally:
        report_file.close()
