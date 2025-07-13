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
model = "ChatGPT 4o"
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_point_cloud_rel.csv"
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\plots\paper_plots\4o"
os.makedirs(save_dir, exist_ok=True)

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

    # Only use available agents for this geometry
    available_agents = df[df["Geometry"] == df_geom_name]["Agent"].unique()
    agents_in_plot = [a for a in agent_order if a in available_agents]

    if not agents_in_plot:
        print(f"[INFO] Skipping geometry '{geom}' – no matching agents found.")
        continue

    subset = df[df["Geometry"] == df_geom_name].set_index("Agent").reindex(agents_in_plot).reset_index()

    if subset.isnull().values.any():
        print(f"[WARN] Missing values for geometry '{geom}':\n", subset)

    # Bar positions
    ind = range(len(agents_in_plot))
    width = 0.25

    # === RELATIVE ERROR PLOT ===
    try:
        clgd_rel = subset["CLGD Rel. Error (×GT-GT)"]
        chamfer_rel = subset["Chamfer Rel. Error (×GT-GT)"]
        hausdorff_rel = subset["Hausdorff Rel. Error (×GT-GT)"]

        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        ax.bar([i - width for i in ind], clgd_rel, width, label='CLGD', color='goldenrod')
        ax.bar(ind, chamfer_rel, width, label='Chamfer', color='cornflowerblue')
        ax.bar([i + width for i in ind], hausdorff_rel, width, label='Hausdorff', color='mediumseagreen')

        ax.set_ylabel('Relative Error (×GT-GT)')
        ax.set_title(f'{geom} – Relative Error ({model})')
        ax.set_xticks(ind)
        ax.set_xticklabels(agents_in_plot)
        ax.set_ylim(0, max(clgd_rel.max(), chamfer_rel.max(), hausdorff_rel.max()) * 1.1)
        ax.legend(loc='upper right', frameon=False)
        plt.tight_layout()

        fig.savefig(os.path.join(save_dir, f"{geom}_rel_error.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(save_dir, f"{geom}_rel_error.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)
    except KeyError as e:
        print(f"[ERROR] Missing column in geometry '{geom}': {e}")
