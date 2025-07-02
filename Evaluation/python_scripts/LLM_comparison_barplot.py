import pandas as pd
import matplotlib.pyplot as plt
import os

# Fixed model order for consistent colors and legend
model_list = ["Opus 4", "Gemini", "GPT-o3", "GPT-4o", "DeepSeek"]
legend_labels = [
    "Claude Opus 4",
    "Gemini 2.5 Pro",
    "ChatGPT o3",
    "ChatGPT 4o",
    "deepseek-chat"
]

# Filepaths and model names (must match model_list order)
files = {
    "Opus 4": r"_Anthropic_Claude_Opus_4\Results\opus_4_by_agents_geometry.csv",
    "Gemini": r"_Google_Gemini_2.5_Pro\Results\gemini_by_agents_geometry.csv",
    "DeepSeek": r"_DeepSeek_DeepSeek_chat\Results\deepseek_by_agents_geometry.csv",
    "GPT-4o": r"_OpenAI_ChatGPT_4o\Results\4o_by_agents_geometry.csv",
    "GPT-o3": r"_OpenAI_ChatGPT_o3\Results\o3_by_agents_geometry.csv"
}

# Assign fixed colors
colors = plt.cm.tab10.colors
model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_list)}

# Read and combine data
dfs = []
for model in model_list:
    path = files[model]
    df = pd.read_csv(path)
    df["Model"] = model
    dfs.append(df)
all_data = pd.concat(dfs, ignore_index=True)

def plot_iou(agent_mode, ax):
    data = all_data[all_data["AgentMode"] == agent_mode]
    object_groups = ["box", "u_profile", "right_angle", "toycar_new"]  # specified order
    bar_width = 0.15
    x = range(len(object_groups))

    for idx, obj in enumerate(object_groups):
        # Get (model, IoU_mean) for this object group
        group_data = []
        for model in model_list:
            val = data[(data["Model"] == model) & (data["Object Group"] == obj)]["IoU_mean"]
            group_data.append((model, val.values[0] if not val.empty else 0))
        # Sort by IoU_mean descending
        group_data_sorted = sorted(group_data, key=lambda x: x[1], reverse=True)
        # Plot bars for this group, sorted
        for i, (model, iou) in enumerate(group_data_sorted):
            ax.bar(idx + i*bar_width, iou, width=bar_width, color=model_color_map[model], label=model if idx == 0 else "")

    # Set x-ticks in the center of each group
    ax.set_xticks([idx + 2*bar_width for idx in x])
    ax.set_xticklabels(object_groups)
    ax.set_ylabel("IoU score")
    ax.set_title(f"IoU by Geometry and different LLMs ({agent_mode})")
    # Fixed legend order and color, with custom labels
    handles = [plt.Rectangle((0,0),1,1, color=model_color_map[model]) for model in model_list]
    ax.legend(handles, legend_labels, bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)

# Plot and save ZeroShot
fig1, ax1 = plt.subplots(figsize=(9, 6))
plot_iou("ZeroShot", ax1)
plt.tight_layout()
plt.savefig(r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\LLM_comparison\LLM_comparison_barplot_ZeroShot_sorted.png", bbox_inches="tight")
plt.close(fig1)

# Plot and save StepPlanning
fig2, ax2 = plt.subplots(figsize=(9, 6))
plot_iou("StepPlanning", ax2)
plt.tight_layout()
plt.savefig(r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\LLM_comparison\LLM_comparison_barplot_StepPlanning_sorted.png", bbox_inches="tight")
plt.close(fig2)