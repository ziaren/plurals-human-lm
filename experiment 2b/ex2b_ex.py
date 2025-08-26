import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm.auto import tqdm

# --------- Config ----------
INPUT_CSV  = "poortman_verb_types_with_sentences.csv"  # update path if needed
OUTPUT_CSV = "ex2b_results.csv"
BATCH_SIZE = 16
# ---------------------------

def pick_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1

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
    # Locate entailment index
    id2label = mdl.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}
    entailment_id = label2id.get("entailment")
    if entailment_id is None:
        for k, v in id2label.items():
            if "entail" in v.lower():
                entailment_id = k
                break
    return clf, entailment_id

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def entailment_probs(nli_pipe, entailment_idx, premises, hypotheses, batch_size=BATCH_SIZE):
    probs = []
    for batch_pairs in tqdm(list(batched(list(zip(premises, hypotheses)), batch_size)), desc="Scoring", leave=False):
        inputs = [{"text": p, "text_pair": h} for p, h in batch_pairs]
        out = nli_pipe(inputs)
        for scores_for_item in out:
            try:
                ent_score = scores_for_item[entailment_idx]["score"]
            except Exception:
                ent_score = max([d["score"] for d in scores_for_item if "entail" in d["label"].lower()])
            probs.append(float(ent_score))
    return probs

def main():
    df = pd.read_csv(INPUT_CSV)
    if "Sentence" not in df.columns or "Sentence_symmetric" not in df.columns:
        raise ValueError("CSV must contain 'Sentence' and 'Sentence_symmetric' columns.")

    premises  = df["Sentence"].astype(str).tolist()
    h_symmetric = df["Sentence_symmetric"].astype(str).tolist()

    device = pick_device()
    print(f"Using device: {device}")

    # Load models
    bart_pipe, bart_ent_id = load_nli_pipeline("facebook/bart-large-mnli", device)
    roberta_pipe, roberta_ent_id = load_nli_pipeline("roberta-large-mnli", device)

    # Score pairs
    print("Scoring BART: Sentence ⇒ Sentence_symmetric ...")
    bart_symmetric = entailment_probs(bart_pipe, bart_ent_id, premises, h_symmetric, BATCH_SIZE)

    print("Scoring RoBERTa: Sentence ⇒ Sentence_symmetric ...")
    roberta_symmetric = entailment_probs(roberta_pipe, roberta_ent_id, premises, h_symmetric, BATCH_SIZE)

    # Save
    df["bart_symmetric"] = bart_symmetric
    df["roberta_symmetric"] = roberta_symmetric
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(df[["Sentence", "Sentence_symmetric", "bart_symmetric", "roberta_symmetric"]].head())

if __name__ == "__main__":
    main()