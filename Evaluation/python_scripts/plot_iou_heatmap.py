import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
model = "ChatGPT o3"
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_o3\Results\o3_by_agents_geometry.csv"
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\o3"
os.makedirs(save_dir, exist_ok=True)

# Display order and mappings
geometry_order = ["box", "u-profile", "right-angle", "toy-car"]
geometry_map = {
    "box": "box",
    "u-profile": "u_profile",
    "right-angle": "right_angle",
    "toy-car": "toycar_new"
}
agent_order = ["ZeroShot", "StepPlanning", "askBack", "visualInspection"]

# === LOAD CSV ===
df = pd.read_csv(csv_path)

# === MAP INTERNAL GEOMETRY NAMES TO DISPLAY NAMES ===
inv_geom_map = {v: k for k, v in geometry_map.items()}
df["Object Group"] = df["Object Group"].map(inv_geom_map)

# === FILTER VALID AGENTS AND GEOMETRIES ===
available_agents = [a for a in agent_order if a in df["AgentMode"].unique()]
available_geoms = [g for g in geometry_order if g in df["Object Group"].unique()]

# === CREATE PIVOT TABLE ===
heatmap_data = df.pivot(index="Object Group", columns="AgentMode", values="IoU_mean")
heatmap_data = heatmap_data.loc[available_geoms, available_agents]

# === PLOT HEATMAP ===
fig, ax = plt.subplots(figsize=(3.2, 2.4))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0.4,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'IoU'},
    annot_kws={"size": 6},
    ax=ax
)

ax.set_title(f"IoU Heatmap by Agent and Geometry ({model})")
ax.set_ylabel("Geometry")
ax.set_xlabel("Agent")
plt.tight_layout()

# === SAVE ===
fig.savefig(os.path.join(save_dir, "IoU_heatmap.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(save_dir, "IoU_heatmap.png"), bbox_inches='tight', dpi=300)
plt.close(fig)
