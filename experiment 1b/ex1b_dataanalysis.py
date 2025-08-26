import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "ex1b_results.csv"   # change path if needed
df = pd.read_csv(file_path)

# Rename Type values
type_mapping = {
    1: "neutral",
    2: "non-symmetric",
    3: "strongly non-symmetric"
}
df["Type"] = df["Type"].map(type_mapping)

# Group by Type and calculate mean for Results_large and Results_small
avg_scores = df.groupby("Type")[["Results_large", "Results_small"]].mean().reset_index()

# Plot
plt.figure(figsize=(8, 6))
bar_width = 0.35
x = range(len(avg_scores))

# Bar plots
plt.bar([i - bar_width/2 for i in x], avg_scores["Results_large"], 
        width=bar_width, label="Results_large", alpha=0.7)
plt.bar([i + bar_width/2 for i in x], avg_scores["Results_small"], 
        width=bar_width, label="Results_small", alpha=0.7)

# Labels and title
plt.xticks(x, avg_scores["Type"], rotation=20)
plt.ylabel("Average Score")
plt.title("Average Results by Type")
plt.ylim(0.7, 0.85)  # zoom y-axis
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()