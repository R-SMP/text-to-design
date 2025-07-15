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
legend_labels = [
    "Claude Opus 4",
    "Gemini 2.5 Pro",
    "ChatGPT o3",
    "ChatGPT 4o",
    "deepseek-chat"
]

files = {
    "Opus 4": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_Anthropic_Claude_Opus_4\Results\opus_4_by_agents_geometry.csv",
    "Gemini": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_Google_Gemini_2.5_Pro\Results\gemini_by_agents_geometry.csv",
    "DeepSeek": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_DeepSeek_DeepSeek_chat\Results\deepseek_by_agents_geometry.csv",
    "GPT-4o": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents_geometry.csv",
    "GPT-o3": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_o3\Results\o3_by_agents_geometry.csv"
}

colors = plt.cm.tab10.colors
model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_list)}

save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\llm_comparison"
os.makedirs(save_dir, exist_ok=True)

# === LOAD & MERGE DATA ===
dfs = []
for model in model_list:
    df = pd.read_csv(files[model])
    df["Model"] = model
    dfs.append(df)
all_data = pd.concat(dfs, ignore_index=True)

# === PLOT FUNCTION ===
def plot_iou(agent_mode, ax):
    data = all_data[all_data["AgentMode"] == agent_mode]
    object_groups = ["box", "u_profile", "right_angle", "toycar_new"]
    bar_width = 0.15
    x = range(len(object_groups))

    for idx, obj in enumerate(object_groups):
        group_data = []
        for model in model_list:
            val = data[(data["Model"] == model) & (data["Object Group"] == obj)]["IoU_mean"]
            group_data.append((model, val.values[0] if not val.empty else 0))
        group_data_sorted = sorted(group_data, key=lambda x: x[1], reverse=True)

        for i, (model, iou) in enumerate(group_data_sorted):
            ax.bar(idx + i * bar_width, iou, width=bar_width, color=model_color_map[model], label=model if idx == 0 else "")

    ax.set_xticks([idx + 2 * bar_width for idx in x])
    ax.set_xticklabels(["box", "u-profile", "right-angle", "toy-car"])
    ax.set_ylabel("IoU")
    ax.set_xlabel("Geometry")
    ax.set_title(f"{agent_mode} – IoU by Geometry")
    ax.set_ylim(0, 1.4)  # Headroom
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Skala limitiert
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_color_map[m]) for m in model_list]
    ax.legend(
    handles,
    legend_labels,
    loc="upper right",
    bbox_to_anchor=(0.99, 0.98),
    ncol=2,
    frameon=False,
    borderaxespad=0.2
)


# === GENERATE PLOTS ===
for agent_mode in ["ZeroShot", "StepPlanning"]:
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    plot_iou(agent_mode, ax)
    plt.tight_layout()
    fname_base = f"LLM_comparison_barplot_{agent_mode}"
    fig.savefig(os.path.join(save_dir, f"{fname_base}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, f"{fname_base}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
