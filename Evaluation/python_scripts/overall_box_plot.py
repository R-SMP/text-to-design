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
model_list = ["Opus 4", "Gemini", "GPT-o3", "GPT-4o", "DeepSeek"]
display_names = {
    "Opus 4": "Claude Opus 4",
    "Gemini": "Gemini 2.5 Pro",
    "GPT-o3": "ChatGPT o3",
    "GPT-4o": "ChatGPT 4o",
    "DeepSeek": "deepseek-chat"
}
files = {
    "Opus 4": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_Anthropic_Claude_Opus_4\Results\opus_4_all.csv",
    "Gemini": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_Google_Gemini_2.5_Pro\Results\gemini_all.csv",
    "DeepSeek": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_DeepSeek_DeepSeek_chat\Results\deepseek_all.csv",
    "GPT-4o": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_all.csv",
    "GPT-o3": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_o3\Results\o3_all.csv"
}
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\llm_comparison"
os.makedirs(save_dir, exist_ok=True)

# === COLOR SETUP ===
colors = plt.cm.tab10.colors
model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_list)}

# === LOAD IoU VALUES ===
iou_data = []
for model, path in files.items():
    df = pd.read_csv(path)
    iou_mean = df[df["Metric"] == "IoU"]["Mean"].values[0]
    iou_data.append((model, iou_mean))

# === SORTING ===
iou_data.sort(key=lambda x: x[1], reverse=True)
models, iou_means = zip(*iou_data)
bar_colors = [model_color_map[model] for model in models]
display_labels = [display_names[model] for model in models]

# === PLOT ===
fig, ax = plt.subplots(figsize=(3.2, 2.4))
bars = ax.bar(display_labels, iou_means, color=bar_colors)
ax.set_ylabel("IoU (Mean)")
ax.set_xlabel("LLM")
ax.set_title("Overall IoU Performance")
ax.set_ylim(0, 1.5)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# === BAR LABELS ===
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=6)

# === LEGEND ===
handles = [plt.Rectangle((0, 0), 1, 1, color=model_color_map[m]) for m in models]
ax.legend(
    handles,
    [display_names[m] for m in models],
    loc="upper right",
    bbox_to_anchor=(0.99, 0.98),
    ncol=2,
    frameon=False,
    borderaxespad=0.2
)

# === SAVE ===
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "LLM_overall_IoU_barplot.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(save_dir, "LLM_overall_IoU_barplot.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
