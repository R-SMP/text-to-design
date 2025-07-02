import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents.csv"
df = pd.read_csv(csv_path)

# Filter for StepPlanning, askBack, and visualInspection agents
agents_of_interest = ["StepPlanning", "askBack", "visualInspection"]
df_filtered = df[df["AgentMode"].isin(agents_of_interest)]

# Bar plot without error bars
plt.figure(figsize=(7, 5))
plt.bar(
    df_filtered["AgentMode"],
    df_filtered["IoU_mean"],
    color=["#ff7f0e", "#1f77b4", "#2ca02c"],
    alpha=0.8
)
plt.ylabel("IoU")
plt.ylim(0, 1.1)
plt.title("IoU Comparison: StepPlanning vs. askBack vs. visualInspection")
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate values on bars
for idx, row in enumerate(df_filtered.itertuples()):
    plt.text(
        idx,
        row.IoU_mean + 0.03,
        f"{row.IoU_mean:.2f}",
        ha='center',
        fontsize=10
    )

plt.tight_layout()

# Save the plot
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\4o"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "IoU_StepPlanning_askBack_visualInspection.png")
plt.savefig(save_path)
plt.show()