import pandas as pd
import os

total_mean = True
#set to true what you want to consider
point_cloud_and_mesh_base = True  # Set to True if you want to consider both point cloud and mesh base results
only_mesh_base = False  # Set to True if you want to consider only mesh base results

# Set to True what u want to subdivide the results by
subvide_object_name_and_agent_mode = True  # Set to True if you want to subdivide by Object Name and AgentMode
subvide_agent_mode =  False  # Set to True if you want to subdivide by AgentMode
only_object_groupe = False  # Set to True if you want to consider only the Object Group


csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\_OpenAI_ChatGPT_4o\results_testing_phase_4o_stepPlanning_zeroShot.csv"

if total_mean == False:
    #this part of the script is for the case where both point cloud and mesh base results are considered
    if point_cloud_and_mesh_base == True:
        # This part of the script subdivides the results of the testing phase 01 by AgentMode and Object Name
        if subvide_object_name_and_agent_mode == True:
            # Path to your input CSV file
            # Load the CSV file
            df = pd.read_csv(csv_path)

            df['Object Group'] = df['Object Name'].str.extract(r'^(.+?)(?:_[a-z])?$')[0]


            # Metrics to calculate
            metrics = ['CLGD', 'Chamfer Distance', 'Hausdorff Distance', 'IoU', 'IoGT']

            # Group by both 'Object Name' and 'AgentMode'
            summary_df = df.groupby(['Object Group', 'AgentMode'])[metrics].agg(['mean', 'std'])

            # Flatten MultiIndex columns
            summary_df.columns = ['_'.join(col) for col in summary_df.columns]

            # Reset index for cleaner output
            summary_df = summary_df.reset_index()

            # Round all numeric values to 5 decimal places
            summary_df = summary_df.round(5)

            # Output file path
            output_path = os.path.join(os.path.dirname(csv_path), 'results_testing_phase01_avg_std_by_agent_rounded.csv')

            # Save to CSV
            summary_df.to_csv(output_path, index=False)

            print(f"Rounded summary saved to: {output_path}")





        # This part of the script subdivides the results of the testing phase 01 only by AgentMode
        if subvide_agent_mode == True:
            # Path to your input CSV file
            # Load the CSV file
            df = pd.read_csv(csv_path)

            df['Object Group'] = df['Object Name'].str.extract(r'^(.+?)(?:_[a-z])?$')[0]

            # Metrics to process
            metrics = ['CLGD', 'Chamfer Distance', 'Hausdorff Distance', 'IoU', 'IoGT']

            # Group by 'AgentMode' only
            summary_df = df.groupby('AgentMode')[metrics].agg(['mean', 'std'])

            # Flatten the MultiIndex columns
            summary_df.columns = ['_'.join(col) for col in summary_df.columns]

            # Reset index to make 'AgentMode' a column
            summary_df = summary_df.reset_index()

            # Round to 5 decimal places
            summary_df = summary_df.round(5)

            # Define output path
            output_path = os.path.join(os.path.dirname(csv_path), 'results_testing_phase01_avg_std_by_agent_only.csv')

            # Save to CSV
            summary_df.to_csv(output_path, index=False)

            print(f"Summary by AgentMode saved to: {output_path}")
        if only_object_groupe == True:
            # Path to your input CSV file
            # Load the CSV file
            df = pd.read_csv(csv_path)

            df['Object Group'] = df['Object Name'].str.extract(r'^(.+?)(?:_[a-z])?$')[0]

            # Metrics to process
            metrics = ['CLGD', 'Chamfer Distance', 'Hausdorff Distance', 'IoU', 'IoGT']

            # Group by 'Object Group' only
            summary_df = df.groupby('Object Group')[metrics].agg(['mean', 'std'])

            # Flatten the MultiIndex columns
            summary_df.columns = ['_'.join(col) for col in summary_df.columns]

            # Reset index to make 'Object Group' a column
            summary_df = summary_df.reset_index()

            # Round to 5 decimal places
            summary_df = summary_df.round(5)

            # Define output path
            output_path = os.path.join(os.path.dirname(csv_path), 'results_testing_phase01_avg_std_by_object_group.csv')

            # Save to CSV
            summary_df.to_csv(output_path, index=False)

            print(f"Summary by Object Group saved to: {output_path}")


    # This part of the script is for the case where only mesh base results are considered
    if only_mesh_base == True:
        # This part of the script subdivides the results of the testing phase 01 by AgentMode and Object Name
        if subvide_object_name_and_agent_mode == True:
            # Path to your input CSV file
                        # Load the CSV file
            df = pd.read_csv(csv_path)

            df['Object Group'] = df['Object Name'].str.extract(r'^(.+?)(?:_[a-z])?$')[0]


            # Metrics to calculate
            metrics = ['IoU', 'IoGT']

            # Group by both 'Object Name' and 'AgentMode'
            summary_df = df.groupby(['Object Group', 'AgentMode'])[metrics].agg(['mean', 'std'])

            # Flatten MultiIndex columns
            summary_df.columns = ['_'.join(col) for col in summary_df.columns]

            # Reset index for cleaner output
            summary_df = summary_df.reset_index()

            # Round all numeric values to 5 decimal places
            summary_df = summary_df.round(5)

            # Output file path
            output_path = os.path.join(os.path.dirname(csv_path), 'results_testing_phase01_avg_std_by_agent_rounded_mesh_based.csv')

            # Save to CSV
            summary_df.to_csv(output_path, index=False)

            print(f"Rounded summary saved to: {output_path}")





        # This part of the script subdivides the results of the testing phase 01 only by AgentMode
        if subvide_agent_mode == True:
            # Path to your input CSV file
                        # Load the CSV file
            df = pd.read_csv(csv_path)

            df['Object Group'] = df['Object Name'].str.extract(r'^(.+?)(?:_[a-z])?$')[0]


            # Metrics to process
            metrics = ['IoU', 'IoGT']

            # Group by 'AgentMode' only
            summary_df = df.groupby('AgentMode')[metrics].agg(['mean', 'std'])

            # Flatten the MultiIndex columns
            summary_df.columns = ['_'.join(col) for col in summary_df.columns]

            # Reset index to make 'AgentMode' a column
            summary_df = summary_df.reset_index()

            # Round to 5 decimal places
            summary_df = summary_df.round(5)

            # Define output path
            output_path = os.path.join(os.path.dirname(csv_path), 'results_testing_phase01_avg_std_by_agent_only_mesh_based.csv')

            # Save to CSV
            summary_df.to_csv(output_path, index=False)

            print(f"Summary by AgentMode saved to: {output_path}")



        if only_object_groupe == True:
            # Path to your input CSV file
                        # Load the CSV file
            df = pd.read_csv(csv_path)

            df['Object Group'] = df['Object Name'].str.extract(r'^(.+?)(?:_[a-z])?$')[0]

            # Metrics to process
            metrics = ['IoU', 'IoGT']

            # Group by 'Object Group' only
            summary_df = df.groupby('Object Group')[metrics].agg(['mean', 'std'])

            # Flatten the MultiIndex columns
            summary_df.columns = ['_'.join(col) for col in summary_df.columns]

            # Reset index to make 'Object Group' a column
            summary_df = summary_df.reset_index()

            # Round to 5 decimal places
            summary_df = summary_df.round(5)

            # Define output path
            output_path = os.path.join(os.path.dirname(csv_path), 'results_testing_phase01_avg_std_by_object_group_mesh_based.csv')

            # Save to CSV
            summary_df.to_csv(output_path, index=False)

            print(f"Summary by Object Group saved to: {output_path}")

else:
    #here overall mean and std of the results are calculated
    # Path to your input CSV file
        # Load the CSV file 
    df = pd.read_csv(csv_path)
    
    metrics = ['CLGD', 'Chamfer Distance', 'Hausdorff Distance', 'IoU', 'IoGT']
    # Calculate overall mean and std for each metric
    overall_mean = df[metrics].mean().round(5)
    overall_std = df[metrics].std().round(5)
    # Combine mean and std into a single DataFrame
    overall_summary = pd.DataFrame({
        'Metric': metrics,
        'Mean': overall_mean,
        'Std': overall_std
    })
    # Define output path
    output_path = os.path.join(os.path.dirname(csv_path), 'overall_results_testing_phase01_avg_std.csv')
    # Save to CSV
    overall_summary.to_csv(output_path, index=False)
    print(f"Overall summary saved to: {output_path}")   
