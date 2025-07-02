import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV file (already contains the mean)
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\Run-time\mean_run_time_by_LLM.csv"
df = pd.read_csv(csv_path)
df.columns = [col.strip() for col in df.columns]
df['LLM model'] = df['LLM model'].str.strip()
df['Run-Time[s]'] = df['Run-Time[s]'].astype(float)

# Set the order and colors to match your image
order = [
    'Gemini 2.5 Pro',
    'ChatGPT o3',
    'Claude Opus 4',
    'ChatGPT 4o',
    'deepseek-chat',
]
colors = [
    '#ff9800',  # Gemini 2.5 Pro (orange)
    '#43a047',  # ChatGPT o3 (green)
    '#1976d2',  # Claude Opus 4 (blue)
    '#d32f2f',  # ChatGPT 4o (red)
    '#9575cd',  # deepseek-chat (purple)
]

# Reindex dataframe to match order
df = df.set_index('LLM model').reindex(order)

# Plot
plt.figure(figsize=(10, 7))
bars = plt.bar(df.index, df['Run-Time[s]'], color=colors)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f'{yval:.1f}',
        ha='center',
        va='bottom',
        fontsize=12,
    )

plt.title('Run-Time mean of zeroShot and stepPlanning by LLM Model (seconds)')
plt.ylabel('Run-Time (s)')
plt.xlabel('LLM Model')
plt.tight_layout()

# Save the plot
save_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\LLM_comparison"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "llm_avg_runtime.png"))