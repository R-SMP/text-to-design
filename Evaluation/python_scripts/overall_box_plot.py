import pandas as pd
import matplotlib.pyplot as plt

# Use absolute paths for each file
files = {
    "Opus 4": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_Anthropic_Claude_Opus_4\Results\opus_4_all.csv",
    "Gemini": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_Google_Gemini_2.5_Pro\Results\gemini_all.csv",
    "DeepSeek": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_DeepSeek_DeepSeek_chat\Results\deepseek_all.csv",
    "GPT-4o": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_all.csv",
    "GPT-o3": r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_o3\Results\o3_all.csv"
}

# Custom display names (same order as keys in files)
display_names = {
    "Opus 4": "Claude Opus 4",
    "Gemini": "Gemini 2.5 Pro",
    "GPT-o3": "ChatGPT o3",
    "GPT-4o": "ChatGPT 4o",
    "DeepSeek": "deepseek-chat"
}

# Assign fixed colors for consistency
model_list = ["Opus 4", "Gemini", "GPT-o3", "GPT-4o", "DeepSeek"]
colors = plt.cm.tab10.colors
model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_list)}

# Read IoU means
iou_data = []
for model, path in files.items():
    df = pd.read_csv(path)
    iou_mean = df[df["Metric"] == "IoU"]["Mean"].values[0]
    iou_data.append((model, iou_mean))

# Sort by IoU descending
iou_data.sort(key=lambda x: x[1], reverse=True)

# Unpack for plotting
models, iou_means = zip(*iou_data)
bar_colors = [model_color_map[model] for model in models]
display_labels = [display_names[model] for model in models]

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(display_labels, iou_means, color=bar_colors)
plt.ylabel("IoU (Mean)")
plt.xlabel("LLM Model")
plt.title("Overall IoU Performance by LLM Model (zeroShot and stepPlanning)")
plt.ylim(0, 1.05)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\LLM_comparison\LLM_overall_IoU_barplot.png", bbox_inches="tight")
plt.show()