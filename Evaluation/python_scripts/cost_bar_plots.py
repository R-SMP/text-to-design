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
llm_order = ["Gemini 2.5 Pro", "ChatGPT o3", "Claude Opus 4", "ChatGPT 4o", "deepseek-chat"]

csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\Cost\LLM,Cost,Input_tokens,Output_tokens.csv"
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\llm_comparison"
os.makedirs(save_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(csv_path, comment='/')
df.columns = [col.strip() for col in df.columns]
df['LLM'] = df['LLM'].str.strip()
df['Cost USD'] = df['Cost USD'].astype(str).str.replace(',', '').astype(float)
df['tokens'] = df['tokens'].astype(str).str.replace(',', '').astype(int)
df['LLM'] = df['LLM'].replace({'DeepSeek': 'deepseek-chat'})

# Order erzwingen
df = df.set_index('LLM').reindex(llm_order).reset_index()

# Farben festlegen (wie im Beispielbild)
llm_color_map = {
    "Gemini 2.5 Pro": "#ff9800",
    "ChatGPT o3": "#43a047",
    "Claude Opus 4": "#1976d2",
    "ChatGPT 4o": "#d32f2f",
    "deepseek-chat": "#9c27b0"
}
bar_colors = [llm_color_map[llm] for llm in df['LLM']]

# USD pro 1M tokens
df['USD per 1M tokens'] = (df['Cost USD'] / df['tokens']) * 1_000_000

# Plot-Funktion
def plot_cost_bar(ax, y_col, ylabel, title, value_fmt):
    bars = ax.bar(df['LLM'], df[y_col], color=bar_colors)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("LLM")
    ax.set_title(title)
    ax.set_xticks(range(len(df['LLM'])))
    ax.set_xticklabels(df['LLM'], rotation=0)
    ax.set_ylim(0, max(df[y_col]) * 1.15)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            value_fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 1),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=6,
            fontweight='normal'
        )

figsize = (3.2, 2.4)

# Plot 1: Cost per LLM
fig, ax = plt.subplots(figsize=figsize)
plot_cost_bar(ax, 'Cost USD', 'Cost (USD)', 'Cost per LLM', "{:.2f}")
plt.tight_layout()
fig.savefig(os.path.join(save_dir, "cost_per_llm.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(save_dir, "cost_per_llm.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

# Plot 2: Tokens per LLM
fig, ax = plt.subplots(figsize=figsize)
plot_cost_bar(ax, 'tokens', 'Tokens', 'Tokens per LLM', "{:,}")
plt.tight_layout()
fig.savefig(os.path.join(save_dir, "tokens_per_llm.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(save_dir, "tokens_per_llm.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

# Plot 3: Price per 1M tokens
fig, ax = plt.subplots(figsize=figsize)
plot_cost_bar(ax, 'USD per 1M tokens', 'USD per 1M tokens', 'Price per 1M Tokens', "{:.2f}")
plt.tight_layout()
fig.savefig(os.path.join(save_dir, "price_per_1M_tokens.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(save_dir, "price_per_1M_tokens.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
