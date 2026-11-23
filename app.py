import os
import html

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# ====================== KONFIGURASI ======================
MIN_USER_RATINGS = 5     # user minimal punya sekian rating eksplisit
MIN_BOOK_RATINGS = 5     # buku minimal dirating sekian user
MIN_KNN_RATINGS = 15     # buku minimal dirating sekian kali untuk model kNN
N_FACTORS = 50           # jumlah faktor laten SVD
RANDOM_STATE = 42
DATA_DIRS = ["data", "."]  # folder yang dicek untuk file CSV

st.set_page_config(page_title="Sistem Rekomendasi Buku", page_icon="📚", layout="wide")


# ====================== LOAD DATA ======================
def find_file(name):
    for d in DATA_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


READ_OPTS = dict(sep=";", encoding="latin-1", on_bad_lines="skip",
                 escapechar="\\", low_memory=False)


@st.cache_data(show_spinner="Memuat & membersihkan data...")
def load_data(books_path, ratings_path):
    books = pd.read_csv(
        books_path, **READ_OPTS,
        usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"],
    ).rename(columns={"Book-Title": "title", "Book-Author": "author",
                      "Year-Of-Publication": "year", "Publisher": "publisher"})
    ratings = pd.read_csv(ratings_path, **READ_OPTS).rename(
        columns={"User-ID": "user_id", "Book-Rating": "rating"})

    for col in ["title", "author", "publisher"]:
        books[col] = books[col].map(lambda x: html.unescape(x) if isinstance(x, str) else x)
    books["ISBN"] = books["ISBN"].astype(str).str.strip().str.upper()
    books = books.drop_duplicates(subset="ISBN").set_index("ISBN")

    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "rating", "ISBN"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(int)
    ratings["ISBN"] = ratings["ISBN"].astype(str).str.strip().str.upper()
    ratings = ratings[ratings["ISBN"].isin(books.index)]
    ratings = ratings.drop_duplicates(subset=["user_id", "ISBN"])

    explicit = ratings[ratings["rating"] > 0].copy()   # hanya rating eksplisit (1-10)
    return books, explicit


def filter_ratings(df, min_user, min_book):
    prev = -1
    while prev != len(df):
        prev = len(df)
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= min_user].index)]
        bc = df["ISBN"].value_counts()
        df = df[df["ISBN"].isin(bc[bc >= min_book].index)]
    return df.reset_index(drop=True)


# ====================== BANGUN MODEL ======================
@st.cache_resource(show_spinner="Melatih model (popularity, kNN, SVD)...")
def build_models(_books, _explicit):
    books, explicit = _books, _explicit
    data = filter_ratings(explicit, MIN_USER_RATINGS, MIN_BOOK_RATINGS)

    test = data.sample(frac=0.2, random_state=RANDOM_STATE)
    train = data.drop(test.index)
    test = test[test["user_id"].isin(train["user_id"]) & test["ISBN"].isin(train["ISBN"])]

    user_ids = np.sort(train["user_id"].unique())
    isbns = np.sort(train["ISBN"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    isbn2idx = {b: i for i, b in enumerate(isbns)}

    rows = train["user_id"].map(user2idx).values
    cols = train["ISBN"].map(isbn2idx).values
    R = csr_matrix((train["rating"].values.astype(float), (rows, cols)),
                   shape=(len(user_ids), len(isbns)))

    # --- Model 1: Popularity (weighted rating) ---
    stats = train.groupby("ISBN").agg(v=("rating", "size"), R=("rating", "mean"))
    C = train["rating"].mean()
    m = stats["v"].quantile(0.90)
    stats["score"] = stats["v"] / (stats["v"] + m) * stats["R"] + m / (stats["v"] + m) * C
    stats = stats.join(books[["title", "author"]])

    # --- Model 2: Item-based kNN (cosine) ---
    knn_items = np.where(R.getnnz(axis=0) >= MIN_KNN_RATINGS)[0]
    knn_pos = {g: p for p, g in enumerate(knn_items)}
    item_user = R[:, knn_items].T.tocsr()
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(item_user)

    # --- Model 3: SVD (mean-centered per user) ---
    user_mean = np.asarray(R.sum(axis=1)).ravel() / np.maximum(R.getnnz(axis=1), 1)
    R_c = R.copy().tocoo()
    R_c.data = R_c.data - user_mean[R_c.row]
    R_c = R_c.tocsr()
    svd = TruncatedSVD(n_components=N_FACTORS, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(R_c)
    item_factors = svd.components_

    # --- Evaluasi (RMSE & MAE) ---
    t_u = test["user_id"].map(user2idx).values
    t_i = test["ISBN"].map(isbn2idx).values
    y = test["rating"].values
    item_mean = np.asarray(R.sum(axis=0)).ravel() / np.maximum(R.getnnz(axis=0), 1)
    preds = {
        "Rata-rata global": np.full(len(y), C),
        "Rata-rata buku": item_mean[t_i],
        f"SVD (k={N_FACTORS})": np.clip(
            (user_factors[t_u] * item_factors[:, t_i].T).sum(axis=1) + user_mean[t_u], 1, 10),
    }
    eval_df = pd.DataFrame({
        "Model": list(preds),
        "RMSE": [float(np.sqrt(np.mean((y - p) ** 2))) for p in preds.values()],
        "MAE": [float(np.mean(np.abs(y - p))) for p in preds.values()],
    }).round(4)

    return dict(
        books=books, train=train, user_ids=user_ids, isbns=isbns, user2idx=user2idx,
        isbn2idx=isbn2idx, R=R, stats=stats, knn=knn, knn_items=knn_items,
        knn_pos=knn_pos, item_user=item_user, user_mean=user_mean,
        user_factors=user_factors, item_factors=item_factors, eval_df=eval_df,
        n_users=len(user_ids), n_books=len(isbns), n_ratings=len(train),
    )


# ====================== FUNGSI REKOMENDASI ======================
def recommend_popular(M, n=10):
    out = M["stats"].sort_values("score", ascending=False).head(n)
    out = out[["title", "author", "v", "R", "score"]].rename(
        columns={"v": "jumlah_rating", "R": "rata2_rating", "score": "weighted_score"})
    return out.round(2).reset_index(drop=True)


def knn_title_options(M):
    """Daftar judul unik (yang ada di model kNN) -> ISBN dengan rating terbanyak."""
    isbn_list = M["isbns"][M["knn_items"]]
    df = M["stats"].loc[isbn_list, ["title", "v"]].dropna(subset=["title"])
    df = df.sort_values("v", ascending=False).drop_duplicates("title")
    return df["title"].tolist(), {t: i for i, t in zip(df.index, df["title"])}


def recommend_similar_books(M, isbn, n=10):
    q = M["knn_pos"][M["isbn2idx"][isbn]]
    k = min(n * 4 + 1, M["item_user"].shape[0])
    dist, ind = M["knn"].kneighbors(M["item_user"][q], n_neighbors=k)
    base_title = M["books"].loc[isbn, "title"]
    res = pd.DataFrame({
        "ISBN": [M["isbns"][M["knn_items"][i]] for i in ind.ravel()[1:]],
        "similarity": (1 - dist.ravel()[1:]).round(3),
    }).join(M["books"][["title", "author"]], on="ISBN")
    res = res[res["title"] != base_title].drop_duplicates("title").head(n)  # buang edisi ganda
    return res[["title", "author", "similarity"]].reset_index(drop=True)


def recommend_for_user(M, user_id, n=10):
    u = M["user2idx"][user_id]
    pred = np.clip(M["user_factors"][u] @ M["item_factors"] + M["user_mean"][u], 1, 10)
    pred[M["R"][u].indices] = -np.inf       # sembunyikan buku yang sudah dirating
    top = np.argsort(-pred)[: n * 3]
    res = pd.DataFrame({"ISBN": M["isbns"][top], "prediksi_rating": pred[top].round(2)})
    res = res.join(M["books"][["title", "author"]], on="ISBN")
    res = res.drop_duplicates("title").head(n)
    return res[["title", "author", "prediksi_rating"]].reset_index(drop=True)


# ====================== UI ======================
st.title("📚 Sistem Rekomendasi Buku")
st.caption("Dataset Book-Crossing · Popularity · Item-based CF (kNN) · Matrix Factorization (SVD)")

books_path = find_file("BX-Books.csv")
ratings_path = find_file("BX-Book-Ratings.csv")

if not books_path or not ratings_path:
    st.error(
        "File **BX-Books.csv** dan **BX-Book-Ratings.csv** tidak ditemukan. "
        "Taruh keduanya di folder `data/` pada repository GitHub (sejajar dengan `app.py`)."
    )
    st.stop()

books, explicit = load_data(books_path, ratings_path)
M = build_models(books, explicit)

c1, c2, c3 = st.columns(3)
c1.metric("User (training)", f"{M['n_users']:,}")
c2.metric("Buku", f"{M['n_books']:,}")
c3.metric("Rating (training)", f"{M['n_ratings']:,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Populer", "📖 Buku Mirip", "👤 Untuk Saya (User)", "📊 Evaluasi"])

with tab1:
    st.subheader("Buku Terpopuler (Weighted Rating)")
    n1 = st.slider("Jumlah rekomendasi", 5, 30, 10, key="n1")
    st.dataframe(recommend_popular(M, n1), use_container_width=True, hide_index=True)
    st.caption("Cocok sebagai fallback untuk user baru (cold-start).")

with tab2:
    st.subheader("Pembaca buku ini juga menyukai…")
    titles, title2isbn = knn_title_options(M)
    default = titles.index("The Da Vinci Code") if "The Da Vinci Code" in titles else 0
    choice = st.selectbox("Pilih / ketik judul buku", titles, index=default)
    n2 = st.slider("Jumlah rekomendasi", 5, 30, 10, key="n2")
    isbn = title2isbn[choice]
    info = M["books"].loc[isbn]
    st.write(f"**Buku acuan:** {info['title']} — {info['author']}")
    st.dataframe(recommend_similar_books(M, isbn, n2),
                 use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Rekomendasi Personal (SVD)")
    top_users = M["train"]["user_id"].value_counts().head(200).index.tolist()
    uid = st.selectbox("Pilih User ID (200 user paling aktif)", top_users)
    manual = st.number_input("…atau masukkan User ID lain (0 = pakai pilihan di atas)",
                             min_value=0, value=0, step=1)
    if manual:
        uid = int(manual)
    n3 = st.slider("Jumlah rekomendasi", 5, 30, 10, key="n3")

    if uid in M["user2idx"]:
        st.markdown("**Buku dengan rating tertinggi dari user ini:**")
        hist = (M["train"][M["train"]["user_id"] == uid]
                .sort_values("rating", ascending=False).head(5)
                .join(M["books"][["title", "author"]], on="ISBN")[["title", "author", "rating"]])
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.markdown("**Rekomendasi untuk user ini:**")
        st.dataframe(recommend_for_user(M, uid, n3),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("User tidak ada di data training → menampilkan rekomendasi populer.")
        st.dataframe(recommend_popular(M, n3), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Evaluasi Prediksi Rating (data test 20%)")
    st.dataframe(M["eval_df"], use_container_width=True, hide_index=True)
    st.caption("Semakin kecil RMSE/MAE semakin baik. SVD mengalahkan kedua baseline.")
