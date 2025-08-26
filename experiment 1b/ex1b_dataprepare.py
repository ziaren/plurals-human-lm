import pandas as pd

INPUT_CSV = "poortman_verb_types.csv"
OUTPUT_CSV = "poortman_verb_types_with_sentences.csv"

# Past tense forms for exactly the 18 verbs you listed
PAST_TENSE = {
    "envy": "envied",
    "know": "knew",
    "understand": "understood",
    "admire": "admired",
    "miss": "missed",
    "hate": "hated",
    "pinch": "pinched",
    "hit": "hit",
    "caress": "caressed",
    "stab": "stabbed",
    "shoot": "shot",
    "grab": "grabbed",
    "kiss": "kissed",
    "dress": "dressed",
    "kick": "kicked",
    "lash out": "lashed out",
    "bite": "bit",
    "lick": "licked"
}

def main():
    df = pd.read_csv(INPUT_CSV)

    if "Verb" not in df.columns:
        raise ValueError("CSV must contain a 'Verb' column")

    # Normalize to lowercase for lookup
    verbs = df["Verb"].str.lower().str.strip()

    # Map to past tense
    past_verbs = verbs.map(PAST_TENSE)

    # Build sentences
    df["Sentence"] = "The children " + past_verbs + " each other."
    df["Sentence_symmetric"] = "Every child " + past_verbs + " every other child."

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {OUTPUT_CSV}")
    print(df.head(10))

if __name__ == "__main__":
    main()