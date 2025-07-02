import pandas as pd
import matplotlib.pyplot as plt
import os

# Model name variable
model = "ChatGPT 4o"  

# Load the CSV file
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents_geometry.csv"
df = pd.read_csv(csv_path)

# List of object groups
object_groups = df["Object Group"].unique()
agents = df["AgentMode"].unique()

# Desired order for AgentMode and Object Group
if model == "ChatGPT 4o":
    agent_order = ["ZeroShot", "StepPlanning", "askBack", "visualInspection"]
else:
    agent_order = ["ZeroShot", "StepPlanning"]

object_order = ["box", "u_profile", "right_angle", "toycar_new"]

# For each object group in the desired order, plot IoU and IoGT for all agents in the desired order
for obj in object_order:
    subset = df[df["Object Group"] == obj]
    subset = subset.set_index("AgentMode").loc[agent_order].reset_index()
    x = subset["AgentMode"]
    iou = subset["IoU_mean"]
    iogt = subset["IoGT_mean"]

    width = 0.35  # width of the bars
    fig, ax = plt.subplots(figsize=(7, 5))
    ind = range(len(x))

    # Plot bars
    bars_iou = ax.bar([i - width/2 for i in ind], iou, width, label='IoU', color='skyblue')
    bars_iogt = ax.bar([i + width/2 for i in ind], iogt, width, label='IoGT', color='orange')

    # Add values inside the bars (small font)
    for bar in bars_iou:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.5,
            f"{height:.2f}",
            ha='center',
            va='center',
            fontsize=7,
            color='black'
        )
    for bar in bars_iogt:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.5,
            f"{height:.2f}",
            ha='center',
            va='center',
            fontsize=7,
            color='black'
        )

    # Labels and title
    ax.set_ylabel('Score')
    ax.set_title(f'IoU and IoGT by Agent for {obj} ({model})')
    ax.set_xticks(ind)
    ax.set_xticklabels(x)
    ax.set_ylim(0, 1.1)

    # Move legend to upper right outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for legend on the right

    # Save each plot as a PNG file in the specified folder
    save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\4o"
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    save_path = os.path.join(save_dir, f"{obj}_IoU_IoGT.png")
    plt.savefig(save_path, bbox_inches='tight')  # Ensure nothing is cut
    plt.show()