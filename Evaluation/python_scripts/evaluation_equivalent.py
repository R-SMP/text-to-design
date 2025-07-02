import pandas as pd
import os

# Path to the original CSV file (use raw string to handle backslashes)
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\benchmark\results_testing_equivalent.csv"

# Load the CSV file
df = pd.read_csv(csv_path)

# Group by 'Object Name' and calculate mean only
summary_df = df.groupby('Object Name')[['CLGD', 'Chamfer Distance', 'Hausdorff Distance']].mean()

# Reset index to make 'Object Name' a column
summary_df = summary_df.reset_index()

# Define output path
output_path = os.path.join(os.path.dirname(csv_path), 'results_equivalent_avg.csv')

# Save to CSV
summary_df.to_csv(output_path, index=False)

print(f"Averaged results saved to: {output_path}")
