import pandas as pd
import matplotlib.pyplot as plt
import os

# === GLOBAL CONFIG ===
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300
})

# === SETTINGS ===
model = "ChatGPT 4o"
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents.csv"
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\4o"
os.makedirs(save_dir, exist_ok=True)

# === DATA FILTERING ===
agents_of_interest = ["StepPlanning", "askBack", "visualInspection"]
df = pd.read_csv(csv_path)
df_filtered = df[df["AgentMode"].isin(agents_of_interest)]
df_filtered = df_filtered.set_index("AgentMode").loc[agents_of_interest].reset_index()

# === PLOT ===
fig, ax = plt.subplots(figsize=(3.2, 2.4))
colors = ["orange", "#1f77b4", "#2ca02c"]
bars = ax.bar(
    df_filtered["AgentMode"],
    df_filtered["IoU_mean"],
    color=colors,
    alpha=0.8
)

ax.set_ylabel("IoU")
ax.set_xlabel("Agent")
ax.set_title(f"IoU Comparison ({model})")
ax.set_ylim(0, 1.1)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.tick_params(axis='both', length=2)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    label_y = min(height + 0.03, 1.0)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=6,
        color='black'
    )

plt.tight_layout()

# === SAVE ===
filename_base = "IoU_StepPlanning_vs_askBack_vs_visualInspection"
fig.savefig(os.path.join(save_dir, f"{filename_base}.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(save_dir, f"{filename_base}.png"), bbox_inches='tight', dpi=300)
plt.close(fig)
