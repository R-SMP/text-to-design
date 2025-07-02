import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Model name variable
model = "ChatGPT 4o"  # Change this to "ChatGPT 4o" if needed

# Load the CSV file
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents_geometry.csv"
df = pd.read_csv(csv_path)

# Desired order for AgentMode columns and Object Group rows
if model == "ChatGPT 4o":
    agent_order = ["ZeroShot", "StepPlanning", "askBack", "visualInspection"]
else:
    agent_order = ["ZeroShot", "StepPlanning"]

object_order = ["box", "u_profile", "right_angle", "toycar_new"]

# Pivot and reindex to desired order
heatmap_data = df.pivot(index="Object Group", columns="AgentMode", values="IoU_mean")
heatmap_data = heatmap_data.loc[object_order, agent_order]

# Plot the heatmap with scale from 0.4 to 1
plt.figure(figsize=(8, 5))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0.4,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'IoU'}
)
plt.title(f"IoU Heatmap by Agent and Geometry ({model})")
plt.ylabel("Geometry")
plt.xlabel("Agent")
plt.tight_layout()

# Save the heatmap
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\4o"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "IoU_heatmap.png")
plt.savefig(save_path, bbox_inches='tight')  # Ensure nothing is cut
plt.show()