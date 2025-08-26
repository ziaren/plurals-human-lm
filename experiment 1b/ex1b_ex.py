import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# --- Config ---
INPUT_CSV = "poortman_verb_types_with_sentences.csv"
OUTPUT_CSV = "ex1b_results.csv"
MODEL_LARGE = "text-embedding-3-large"
MODEL_SMALL = "text-embedding-3-small"

# --- Setup API ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No API key found. Please set OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_embeddings(model: str, texts: list[str]) -> np.ndarray:
    """Fetch embeddings for a list of texts using OpenAI API."""
    res = client.embeddings.create(model=model, input=texts)
    emb_list = [np.array(d.embedding, dtype=np.float32) for d in res.data]
    return np.vstack(emb_list)

def main():
    df = pd.read_csv(INPUT_CSV)

    if not {"Sentence", "Sentence_symmetric"}.issubset(df.columns):
        raise ValueError("CSV must contain 'Sentence' and 'Sentence_symmetric' columns")

    base_sentences = df["Sentence"].astype(str).tolist()
    symm_sentences = df["Sentence_symmetric"].astype(str).tolist()

    # --- Large embeddings
    base_large = get_embeddings(MODEL_LARGE, base_sentences)
    symm_large = get_embeddings(MODEL_LARGE, symm_sentences)
    results_large = [cosine_sim(a, b) for a, b in zip(base_large, symm_large)]

    # --- Small embeddings
    base_small = get_embeddings(MODEL_SMALL, base_sentences)
    symm_small = get_embeddings(MODEL_SMALL, symm_sentences)
    results_small = [cosine_sim(a, b) for a, b in zip(base_small, symm_small)]

    # Store results
    df["Results_large"] = results_large
    df["Results_small"] = results_small

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved results to {OUTPUT_CSV}")
    print(df[["Verb", "Results_large", "Results_small"]].head())

if __name__ == "__main__":
    main()