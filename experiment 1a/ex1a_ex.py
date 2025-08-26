# ex1_cosine_similarity.py
import numpy as np
import pandas as pd
import os, json
from dotenv import load_dotenv
from openai import OpenAI

# 1. Try loading from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. If not found, try config.json
if not api_key and os.path.exists("config.json"):
    with open("config.json") as f:
        api_key = json.load(f).get("OPENAI_API_KEY")

# 3. Final check
if not api_key:
    raise ValueError("❌ No API key found. Please set OPENAI_API_KEY in .env or config.json")

# 4. Initialize client
client = OpenAI(api_key=api_key)

# --- Config ---
INPUT_CSV = "scontras_with_variants.csv"
OUTPUT_CSV = "ex1_results.csv"
MODEL_LARGE = "text-embedding-3-large"
MODEL_SMALL = "text-embedding-3-small"

# --- Setup API ---
# Uses OPENAI_API_KEY from environment or .env file
load_dotenv()
from openai import OpenAI  # pip install openai
client = OpenAI()  # expects OPENAI_API_KEY to be set

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)

def get_embeddings(model: str, texts: list[str]) -> np.ndarray:
    """
    Calls the OpenAI Embeddings API once for a list of texts and
    returns an array shaped (len(texts), dim).
    """
    # The API accepts a list for "input", returning one embedding per item.
    res = client.embeddings.create(model=model, input=texts)
    # Preserve the input order:
    emb_list = [np.array(d.embedding, dtype=np.float32) for d in res.data]
    return np.vstack(emb_list)

def ensure_columns(df: pd.DataFrame):
    required = ["Sentence", "Sentence_collective", "Sentence_distributive"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

def main():
    # 1) Load data and sanity-check columns
    df = pd.read_csv(INPUT_CSV)
    ensure_columns(df)

    base_texts = df["Sentence"].fillna("").astype(str).tolist()
    coll_texts = df["Sentence_collective"].fillna("").astype(str).tolist()
    dist_texts = df["Sentence_distributive"].fillna("").astype(str).tolist()

    # 2) Embeddings for LARGE model
    base_large = get_embeddings(MODEL_LARGE, base_texts)
    coll_large = get_embeddings(MODEL_LARGE, coll_texts)
    dist_large = get_embeddings(MODEL_LARGE, dist_texts)

    # 3) Embeddings for SMALL model
    base_small = get_embeddings(MODEL_SMALL, base_texts)
    coll_small = get_embeddings(MODEL_SMALL, coll_texts)
    dist_small = get_embeddings(MODEL_SMALL, dist_texts)

    # 4) Cosine similarities (pairwise by row)
    collective_large = [cosine_sim(b, c) for b, c in zip(base_large, coll_large)]
    collective_small = [cosine_sim(b, c) for b, c in zip(base_small, coll_small)]
    distributive_large = [cosine_sim(b, d) for b, d in zip(base_large, dist_large)]
    distributive_small = [cosine_sim(b, d) for b, d in zip(base_small, dist_small)]

    # 5) Assemble output DataFrame
    out = df.copy()
    out["collective_large"] = collective_large
    out["collective_small"] = collective_small
    out["distributive_large"] = distributive_large
    out["distributive_small"] = distributive_small

    # 6) Save
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved: {OUTPUT_CSV}")
    # Optional quick peek:
    print(out[["Sentence","collective_large","collective_small","distributive_large","distributive_small"]].head())

if __name__ == "__main__":
    main()