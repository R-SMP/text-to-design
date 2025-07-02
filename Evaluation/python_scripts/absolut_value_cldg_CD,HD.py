import pandas as pd
import matplotlib.pyplot as plt
import os
model = "ChatGPT 4o"
# Set the path to the CSV file and the directory for saving plots
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_by_agents_geometry.csv"
plot_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\4o"
os.makedirs(plot_dir, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

# Rename columns for easier access
df = df.rename(columns={
    'Object Group': 'Geometry',
    'AgentMode': 'Agent',
    'CLGD_mean': 'CLGD',
    'Chamfer Distance_mean': 'Chamfer Distance',
    'Hausdorff Distance_mean': 'Hausdorff Distance'
})

# Desired order for plotting
agent_order = ["ZeroShot", "StepPlanning", "askBack", "visualInspection"]
geometry_order = ["box", "u_profile", "right_angle", "toycar_new"]

for geom in geometry_order:
    subset = df[df['Geometry'] == geom]
    # Ensure correct agent order and fill missing agents with zeros
    subset = subset.set_index('Agent').reindex(agent_order).fillna(0).reset_index()
    x = subset['Agent']
    clgd = subset['CLGD']
    chamfer = subset['Chamfer Distance']
    hausdorff = subset['Hausdorff Distance']

    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ind = range(len(x))

    # Plot bars
    ax.bar([i - width for i in ind], clgd, width, label='CLGD', color='goldenrod')
    ax.bar(ind, chamfer, width, label='Chamfer Distance', color='cornflowerblue')
    ax.bar([i + width for i in ind], hausdorff, width, label='Hausdorff Distance', color='mediumseagreen')

    ax.set_ylabel('Mean Absolute Value')
    ax.set_title(f'CLGD, Chamfer & Hausdorff Distance (mean) by Agent for {geom} ({model})')
    ax.set_xticks(ind)
    ax.set_xticklabels(x)
    ax.set_ylim(0, max(clgd.max(), chamfer.max(), hausdorff.max()) * 1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = os.path.join(r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\4o", f"{geom}_CLGD_Chamfer_Hausdorff_Absolute_Mean.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
