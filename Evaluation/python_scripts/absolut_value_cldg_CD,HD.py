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
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design-clean\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents_geometry.csv"
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
df = df.rename(columns={
    'Object Group': 'Geometry',
    'AgentMode': 'Agent',
    'CLGD_mean': 'CLGD',
    'Chamfer Distance_mean': 'Chamfer Distance',
    'Hausdorff Distance_mean': 'Hausdorff Distance'
})

# === PLOT LOOP ===
for geom in geometry_order:
    geom_name = geometry_map[geom]
    subset = df[df['Geometry'] == geom_name]
    subset = subset.set_index('Agent').reindex(agent_order).reset_index()

    if subset.isnull().values.any():
        print(f"[WARN] Missing values for geometry '{geom}':\n", subset)

    clgd = subset['CLGD']
    chamfer = subset['Chamfer Distance']
    hausdorff = subset['Hausdorff Distance']

    width = 0.25
    ind = range(len(agent_order))

    # === ABSOLUTE VALUE PLOT ===
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.bar([i - width for i in ind], clgd, width, label='CLGD', color='goldenrod')
    ax.bar(ind, chamfer, width, label='Chamfer', color='cornflowerblue')
    ax.bar([i + width for i in ind], hausdorff, width, label='Hausdorff', color='mediumseagreen')

    ax.set_ylabel('Mean Absolute Value')
    ax.set_title(f'{geom} – Distance Metrics (mean, {model})')
    ax.set_xticks(ind)
    ax.set_xticklabels(agent_order)
    ax.set_ylim(0, max(clgd.max(), chamfer.max(), hausdorff.max()) * 1.1)
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()

    fig.savefig(os.path.join(save_dir, f"{geom}_abs_mean_error.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f"{geom}_abs_mean_error.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
