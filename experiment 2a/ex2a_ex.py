# ex2a_entailment_probs.py
import os
import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm.auto import tqdm

# --------- Config ----------
INPUT_CSV  = "scontras_with_variants.csv"  # change if needed
OUTPUT_CSV = "ex2a_results.csv"
BATCH_SIZE = 16  # adjust to your memory
# ---------------------------

def pick_device():
    """
    Prefer CUDA if available, otherwise Apple MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return 0  # CUDA device index for pipeline
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # transformers pipeline expects "mps" as a string device
        return "mps"
    return -1  # CPU

def load_nli_pipeline(model_name: str, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        task="text-classification",
        model=mdl,
        tokenizer=tok,
        return_all_scores=True,
        function_to_apply="softmax",
        device=device
    )
    # Make a label->index map so we can grab "entailment" robustly
    id2label = mdl.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}
    # Some models name it "ENTAILMENT"; others might use "entailment"
    # We'll just normalize to lowercase and look up
    entailment_id = label2id.get("entailment")
    if entailment_id is None:
        # Fallback: search by substring
        for k, v in id2label.items():
            if "entail" in v.lower():
                entailment_id = k
                break
    if entailment_id is None:
        raise RuntimeError(f"Could not locate 'entailment' label in {model_name} id2label={id2label}")
    return clf, entailment_id

def batched(iterable, n):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def entailment_probs(nli_pipe, entailment_idx: int, premises, hypotheses, batch_size=BATCH_SIZE):
    """
    Compute P(entailment) for each (premise, hypothesis) pair.
    Returns a list of floats in [0,1].
    """
    assert len(premises) == len(hypotheses)
    probs = []
    for batch_pairs in tqdm(list(batched(list(zip(premises, hypotheses)), batch_size)), desc="Scoring", leave=False):
        # The text-classification pipeline accepts sequence pairs either as tuples
        # or dicts {"text": premise, "text_pair": hypothesis}
        inputs = [{"text": p, "text_pair": h} for p, h in batch_pairs]
        out = nli_pipe(inputs)  # list of [ {label, score}, ... ] for each class
        # For return_all_scores=True, each item is a list of dicts sorted by label id
        for scores_for_item in out:
            # We rely on the pipeline preserving id order; to be safe, index by label
            # Build label->score map (lowercase)
            label_to_score = {d["label"].lower(): d["score"] for d in scores_for_item}
            # If we know the entailment index, find the dict with that label; else fallback to name
            # Try index-based retrieval (in case labels are exactly aligned)
            try:
                ent_score = scores_for_item[entailment_idx]["score"]
            except Exception:
                ent_score = label_to_score.get("entailment")
                if ent_score is None:
                    # last resort: pick the max between keys that contain "entail"
                    ent_candidates = [d["score"] for d in scores_for_item if "entail" in d["label"].lower()]
                    ent_score = max(ent_candidates) if ent_candidates else float("nan")
            probs.append(float(ent_score))
    return probs

def main():
    # --- Load data ---
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["Sentence", "Sentence_collective", "Sentence_distributive"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    device = pick_device()
    print(f"Using device: {device}")

    # --- Load models/pipelines ---
    print("Loading facebook/bart-large-mnli ...")
    bart_pipe, bart_ent_id = load_nli_pipeline("facebook/bart-large-mnli", device)
    print("Loading roberta-large-mnli ...")
    roberta_pipe, roberta_ent_id = load_nli_pipeline("roberta-large-mnli", device)

    # --- Prepare pairs ---
    premises  = df["Sentence"].astype(str).tolist()
    h_collect = df["Sentence_collective"].astype(str).tolist()
    h_distr   = df["Sentence_distributive"].astype(str).tolist()

    print("Scoring BART: Sentence ⇒ Sentence_collective ...")
    bart_collective = entailment_probs(bart_pipe, bart_ent_id, premises, h_collect, BATCH_SIZE)

    print("Scoring BART: Sentence ⇒ Sentence_distributive ...")
    bart_distributive = entailment_probs(bart_pipe, bart_ent_id, premises, h_distr, BATCH_SIZE)

    print("Scoring RoBERTa: Sentence ⇒ Sentence_collective ...")
    roberta_collective = entailment_probs(roberta_pipe, roberta_ent_id, premises, h_collect, BATCH_SIZE)

    print("Scoring RoBERTa: Sentence ⇒ Sentence_distributive ...")
    roberta_distributive = entailment_probs(roberta_pipe, roberta_ent_id, premises, h_distr, BATCH_SIZE)

    # --- Store results ---
    df["bart_collective"]     = bart_collective
    df["bart_distributive"]   = bart_distributive
    df["roberta_collective"]  = roberta_collective
    df["roberta_distributive"]= roberta_distributive

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    # Quick sanity print
    print(df[["Sentence","bart_collective","bart_distributive","roberta_collective","roberta_distributive"]].head())

if __name__ == "__main__":
    main()