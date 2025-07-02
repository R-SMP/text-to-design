import pandas as pd
import matplotlib.pyplot as plt
import os

# Model name variable
model = "ChatGPT 4o"

# Load the CSV file
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\Results\4o_point_cloud_rel.csv"
df = pd.read_csv(csv_path)

# List of geometries
geometries = df["Geometry"].unique()

# Desired order for Agent and Geometry
if model == "ChatGPT 4o":
    agent_order = ["ZeroShot", "StepPlanning", "askBack", "visualInspection"]
else:
    agent_order = ["ZeroShot", "StepPlanning"]

geometry_order = ["box", "u_profile", "right_angle", "toycar_new"]

# For each geometry in the desired order, plot CLGD, Chamfer, and Hausdorff Rel. Error for all agents in the desired order
for geom in geometry_order:
    subset = df[df["Geometry"] == geom]
    subset = subset.set_index("Agent").loc[agent_order].reset_index()
    x = subset["Agent"]
    clgd = subset["CLGD Rel. Error (×GT-GT)"]
    chamfer = subset["Chamfer Rel. Error (×GT-GT)"]
    hausdorff = subset["Hausdorff Rel. Error (×GT-GT)"]

    width = 0.25  # width of the bars
    fig, ax = plt.subplots(figsize=(8, 5))
    ind = range(len(x))

    # Plot bars (Relative Error)
    ax.bar([i - width for i in ind], clgd, width, label='CLGD Rel. Error', color='goldenrod')
    ax.bar(ind, chamfer, width, label='Chamfer Rel. Error', color='cornflowerblue')
    ax.bar([i + width for i in ind], hausdorff, width, label='Hausdorff Rel. Error', color='mediumseagreen')

    # Labels and title
    ax.set_ylabel('Relative Error (×GT-GT)')
    ax.set_title(f'CLGD, Chamfer & Hausdorff Rel. Error by Agent for {geom} ({model})')
    ax.set_xticks(ind)
    ax.set_xticklabels(x)
    ax.set_ylim(0, max(clgd.max(), chamfer.max(), hausdorff.max()) * 1.1)

    # Move legend to upper right outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for legend on the right

    # Save each plot as a PNG file in the specified folder
    save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\4o"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{geom}_CLGD_Chamfer_Hausdorff_Rel_Error.png")
    plt.savefig(save_path, bbox_inches='tight')  # Ensure nothing is cut off
    plt.close(fig)
    plt.show()

    # --- NEW: Plot absolute values (Gen-GT) ---
    clgd_abs = subset["CLGD (Gen-GT)"]
    chamfer_abs = subset["Chamfer (Gen-GT)"]
    hausdorff_abs = subset["Hausdorff (Gen-GT)"]

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # Plot bars (Absolute Values)
    ax2.bar([i - width for i in ind], clgd_abs, width, label='CLGD (Gen-GT)', color='goldenrod')
    ax2.bar(ind, chamfer_abs, width, label='Chamfer (Gen-GT)', color='cornflowerblue')
    ax2.bar([i + width for i in ind], hausdorff_abs, width, label='Hausdorff (Gen-GT)', color='mediumseagreen')

    # Labels and title
    ax2.set_ylabel('Absolute Value (Gen-GT)')
    ax2.set_title(f'CLGD, Chamfer & Hausdorff Absolute Value by Agent for {geom} ({model})')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(x)
    ax2.set_ylim(0, max(clgd_abs.max(), chamfer_abs.max(), hausdorff_abs.max()) * 1.1)

    # Move legend to upper right outside the plot
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for legend on the right

    # Save each absolute value plot as a PNG file
    save_path_abs = os.path.join(save_dir, f"{geom}_CLGD_Chamfer_Hausdorff_Absolute_Value.png")
    plt.savefig(save_path_abs, bbox_inches='tight')
    plt.close(fig2)
    plt.show()