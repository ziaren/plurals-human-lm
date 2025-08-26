import pandas as pd

# Load the CSV file
file_path = "scontras_original.csv"   # change path if needed
df = pd.read_csv(file_path)

# Add "together" before 'were' → Sentence_collective
df["Sentence_collective"] = df["Sentence"].str.replace(
    r"\bwere\b", "together were", regex=True
)

# Add "each" before 'were' → Sentence_distributive
df["Sentence_distributive"] = df["Sentence"].str.replace(
    r"\bwere\b", "each were", regex=True
)

# Save updated CSV
df.to_csv("scontras_with_variants.csv", index=False)

print("✅ New file saved as 'scontras_with_variants.csv'")
