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
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\Run-time\mean_run_time_by_LLM.csv"
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\llm_comparison"
os.makedirs(save_dir, exist_ok=True)

order = [
    'Gemini 2.5 Pro',
    'ChatGPT o3',
    'Claude Opus 4',
    'ChatGPT 4o',
    'deepseek-chat',
]
color_map = {
    'Gemini 2.5 Pro': '#ff9800',
    'ChatGPT o3': '#43a047',
    'Claude Opus 4': '#1976d2',
    'ChatGPT 4o': '#d32f2f',
    'deepseek-chat': '#9c27b0'
}

# === LOAD DATA ===
df = pd.read_csv(csv_path)
df.columns = [col.strip() for col in df.columns]
df['LLM model'] = df['LLM model'].str.strip()
df['Run-Time[s]'] = df['Run-Time[s]'].astype(float)

# Order erzwingen
df = df.set_index('LLM model').reindex(order).reset_index()
bar_colors = [color_map[m] for m in df['LLM model']]

# === PLOT ===
figsize = (3.2, 2.4)
fig, ax = plt.subplots(figsize=figsize)
bars = ax.bar(df['LLM model'], df['Run-Time[s]'], color=bar_colors)

ax.set_xlabel('LLM')
ax.set_ylabel('Run-Time (s)')
ax.set_title('Run-Time mean of zeroShot + stepPlanning by LLM')
ax.set_xticks(range(len(df['LLM model'])))
ax.set_xticklabels(df['LLM model'], rotation=0)

# Mehr Platz oben
ax.set_ylim(0, max(df['Run-Time[s]']) * 1.2)

# Zahlen auf Balken
for bar in bars:
    yval = bar.get_height()
    ax.annotate(
        f'{yval:.1f}',
        xy=(bar.get_x() + bar.get_width() / 2, yval),
        xytext=(0, 1),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=6,
        fontweight='normal'
    )

plt.tight_layout()
fig.savefig(os.path.join(save_dir, "llm_avg_runtime.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(save_dir, "llm_avg_runtime.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
