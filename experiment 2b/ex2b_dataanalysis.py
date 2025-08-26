import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "ex2b_results.csv"
df = pd.read_csv(file_path)

# Calculate average BART and RoBERTa symmetric scores by Type
avg_scores = df.groupby("Type")[["bart_symmetric", "roberta_symmetric"]].mean().reset_index()

# Rename Type values to meaningful labels
type_labels = {1: "neutral", 2: "non-symmetric", 3: "strongly non-symmetric"}
avg_scores["Type"] = avg_scores["Type"].map(type_labels)

# Print summary table
print("Average symmetric scores by type:")
print(avg_scores)

# Plot bar chart
avg_scores.plot(x="Type", kind="bar", rot=0)
plt.title("Average Symmetric Scores by Type")
plt.ylabel("Average Score")
plt.xlabel("Verb Type")
plt.legend(title="Model")
plt.tight_layout()
plt.show()