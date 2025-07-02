import pandas as pd
import matplotlib.pyplot as plt
import os

# Model name variable
model = "ChatGPT 4o"

# Load the CSV file
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents.csv"
df = pd.read_csv(csv_path)

# Desired order for agents
agents_of_interest = ["StepPlanning", "askBack", "visualInspection"]
df_filtered = df[df["AgentMode"].isin(agents_of_interest)]
df_filtered = df_filtered.set_index("AgentMode").loc[agents_of_interest].reset_index()

# Bar plot without error bars
plt.figure(figsize=(6, 5))
colors = ["orange", "#1f77b4", "#2ca02c"]
bars = plt.bar(
    df_filtered["AgentMode"],
    df_filtered["IoU_mean"],
    color=colors,
    alpha=0.8
)
plt.ylabel("IoU")
plt.ylim(0, 1.1)
plt.title(f"IoU Comparison: StepPlanning vs. askBack vs. visualInspection ({model})")
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.03,
        f"{height:.2f}",
        ha='center',
        fontsize=10
    )

plt.tight_layout()

# Save the plot
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\4o"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "IoU_StepPlanning_vs_askBack_vs_visualInspection.png")
plt.savefig(save_path, bbox_inches='tight')  # Ensure nothing is cut
plt.show()