import pandas as pd
import os

print(os.getcwd())

# === 1. Pfade zu den CSV-Dateien ===
gen_gt_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_Google_Gemini_2.5_Pro\Results\gemini_by_agents_geometry.csv"
gt_gt_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\benchmark\equivalent_values.csv"

# === 2. CSV-Dateien laden ===
gen_gt_df = pd.read_csv(gen_gt_path)
gt_gt_df = pd.read_csv(gt_gt_path)

# === 3. Mapping Object Name → Object Group ===
object_name_to_group = {
    "box": "box",
    "right_angle": "right_angle",
    "toycar_new": "toycar_new",
    "u_profile": "u_profile"
}

gt_gt_df["Object Group"] = gt_gt_df["Object Name"].map(object_name_to_group)

# === 4. Mergen auf Object Group ===
merged_df = pd.merge(
    gen_gt_df,
    gt_gt_df[["Object Group", "CLGD", "Chamfer Distance", "Hausdorff Distance"]],
    left_on="Object Group",
    right_on="Object Group",
    how="left"
)

# === 5. Relative Fehler berechnen ===
merged_df["CLGD_rel_error"] = (merged_df["CLGD_mean"] - merged_df["CLGD"]) / merged_df["CLGD"]
merged_df["Chamfer_rel_error"] = (merged_df["Chamfer Distance_mean"] - merged_df["Chamfer Distance"]) / merged_df["Chamfer Distance"]
merged_df["Hausdorff_rel_error"] = (merged_df["Hausdorff Distance_mean"] - merged_df["Hausdorff Distance"]) / merged_df["Hausdorff Distance"]

# === 6. Relevante Spalten auswählen und umbenennen ===
output_df = merged_df[[
    "Object Group", "AgentMode",
    "CLGD_rel_error", "Chamfer_rel_error", "Hausdorff_rel_error"
]]
output_df.columns = [
    "Geometry", "Agent",
    "CLGD Rel. Error (×GT-GT)", "Chamfer Rel. Error (×GT-GT)", "Hausdorff Rel. Error (×GT-GT)"
]

# === 7. Runden und anzeigen ===
output_df = output_df.round(4)
print(output_df)

# === 8. Optional: Als CSV speichern ===
output_df.to_csv(r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_Google_Gemini_2.5_Pro\Results\relative_error_pointcloud_metrics.csv", index=False)
