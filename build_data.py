import pandas as pd, chromadb
from sentence_transformers import SentenceTransformer

N_REVIEWS = 25000
KEEP = ["id", "name", "neighbourhood_cleansed", "property_type", "room_type",
        "accommodates", "bedrooms", "price", "number_of_reviews",
        "review_scores_rating", "review_scores_cleanliness",
        "review_scores_location", "review_scores_value",
        "host_is_superhost", "availability_365"]

# ---- listings ----
L = pd.read_csv("data/listings.csv", usecols=lambda c: c in KEEP)
L = L.rename(columns={"id": "listing_id", "neighbourhood_cleansed": "neighbourhood"})
L["price"] = pd.to_numeric(L["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
L["host_is_superhost"] = L["host_is_superhost"].map({"t": True, "f": False})
L.to_parquet("data/listings.parquet")
print("listings:", L.shape)

# ---- reviews ----
R = pd.read_csv("data/reviews.csv", usecols=["listing_id", "date", "comments"]).dropna()
R["comments"] = (R["comments"].str.replace(r"<br\s*/?>", " ", regex=True)
                 .str.replace(r"\s+", " ", regex=True).str.strip())
R = R[R["comments"].str.len().between(40, 1000)]
R = R[~R["comments"].str.contains("automated posting|canceled this reservation", case=False)]
R = R[R["comments"].map(lambda s: sum(c.isascii() for c in s) / len(s)) > 0.97]  # crude English filter
R = R.sample(min(N_REVIEWS, len(R)), random_state=42)
R["date"] = pd.to_datetime(R["date"])
R = R.merge(L[["listing_id", "neighbourhood", "room_type"]], on="listing_id", how="left").reset_index(drop=True)
R.to_parquet("data/reviews.parquet")
print("reviews:", R.shape)

# ---- embeddings + Chroma ----
model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 128
emb = model.encode(R["comments"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("reviews")
except Exception:
    pass
col = client.create_collection("reviews", metadata={"hnsw:space": "cosine"})
for i in range(0, len(R), 5000):
    ch = R.iloc[i:i + 5000]
    col.add(
        ids=[str(j) for j in ch.index],
        embeddings=emb[i:i + 5000].tolist(),
        documents=ch["comments"].tolist(),
        metadatas=[{"listing_id": int(r.listing_id), "neighbourhood": str(r.neighbourhood),
                    "room_type": str(r.room_type), "date": r.date.strftime("%Y-%m")}
                   for r in ch.itertuples()],
    )
print("done:", col.count())