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
model = "Gemini 2.5 Pro"
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_Google_Gemini_2.5_Pro\Results\gemini_by_agents_geometry.csv"
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\gemini"
os.makedirs(save_dir, exist_ok=True)

# Geometry display order and mapping
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

# === LOOP OVER GEOMETRIES ===
for geom in geometry_order:
    df_geom_name = geometry_map[geom]
    subset = df[df["Object Group"] == df_geom_name]

    available_agents = [a for a in agent_order if a in subset["AgentMode"].unique()]
    if not available_agents:
        print(f"[INFO] Skipping '{geom}' – no valid agents.")
        continue

    subset = subset.set_index("AgentMode").reindex(available_agents).reset_index()

    # Data
    x = subset["AgentMode"]
    iou = subset["IoU_mean"]
    iogt = subset["IoGT_mean"]
    ind = range(len(x))
    width = 0.35

    # Plot
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    bars_iou = ax.bar([i - width/2 for i in ind], iou, width, label='IoU', color='skyblue')
    bars_iogt = ax.bar([i + width/2 for i in ind], iogt, width, label='IoGT', color='orange')

    # Text inside bars
    for bar in bars_iou + bars_iogt:
        height = bar.get_height()
        label_y = min(height * 0.5, height - 0.05, 1.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{height:.2f}",
            ha='center',
            va='center',
            fontsize=6,
            color='black'
        )

    ax.set_ylabel('Score')
    ax.set_xlabel('Agent')
    ax.set_title(f'{geom} – IoU and IoGT ({model})')
    ax.set_xticks(ind)
    ax.set_xticklabels(x)
    ax.set_ylim(0, 1.25)  # Plothöhe
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Achsenskalierung sichtbar nur bis 1.0
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()

    # Save
    filename_base = f"{geom}_IoU_IoGT"
    fig.savefig(os.path.join(save_dir, f"{filename_base}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f"{filename_base}.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
