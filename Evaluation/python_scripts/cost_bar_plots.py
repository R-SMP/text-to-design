import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your CSV file
csv_path = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\Cost\LLM,Cost,Input_tokens,Output_tokens.csv"

# Output directory for plots
output_dir = r"C:\Users\natha\OneDrive\Desktop\OneDrive - ETH Zurich\GitHub\text-to-design\Evaluation\plots\LLM_comparison"
os.makedirs(output_dir, exist_ok=True)

# Read CSV (skip the first line if it has a comment)
df = pd.read_csv(csv_path, comment='/')

# Clean up column names and whitespace
df.columns = [col.strip() for col in df.columns]
df['LLM'] = df['LLM'].str.strip()
df['Cost USD'] = df['Cost USD'].astype(str).str.replace(',', '').astype(float)
df['tokens'] = df['tokens'].astype(str).str.replace(',', '').astype(int)

# Calculate price per token (USD per token), then per million tokens
df['USD per 1M tokens'] = (df['Cost USD'] / df['tokens']) * 1_000_000

# Define color mapping to match overall_box_plot.py
color_map = {
    "Claude Opus 4": "#1f77b4",      # blue
    "Gemini 2.5 Pro": "#ff7f0e",     # orange
    "ChatGPT o3": "#2ca02c",         # green
    "ChatGPT 4o": "#d62728",         # red
    "DeepSeek": "#9467bd",           # purple
    "deepseek-chat": "#9467bd"       # for consistency if needed
}

# Desired LLM order
llm_order = ["Gemini 2.5 Pro", "ChatGPT o3", "Claude Opus 4", "ChatGPT 4o", "deepseek-chat"]

# Ensure LLM names match the color map keys
df['LLM'] = df['LLM'].replace({'DeepSeek': 'deepseek-chat'})

# Reorder DataFrame by desired LLM order
df = df.set_index('LLM').reindex(llm_order).reset_index()

bar_colors = [color_map.get(llm, "#333333") for llm in df['LLM']]

# Plot 1: Cost per LLM
plt.figure(figsize=(8, 4))
bars = plt.bar(df['LLM'], df['Cost USD'], color=bar_colors)
plt.ylabel('Cost (USD)')
plt.title('Cost per LLM')
plt.xticks(rotation=20)
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, -1), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cost_per_llm.png"))
plt.close()

# Plot 2: Tokens per LLM (ordered by desired order)
plt.figure(figsize=(8, 4))
bars = plt.bar(df['LLM'], df['tokens'], color=bar_colors)
plt.ylabel('Tokens')
plt.title('Tokens per LLM')
plt.xticks(rotation=20)
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, -1), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tokens_per_llm.png"))
plt.close()

# Plot 3: Price per 1M tokens (ordered by desired order)
plt.figure(figsize=(8, 4))
bars = plt.bar(df['LLM'], df['USD per 1M tokens'], color=bar_colors)
plt.ylabel('USD per 1M tokens')
plt.title('Price per 1M Tokens by LLM')
plt.xticks(rotation=20)
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, -1), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "price_per_1M_tokens.png"))
plt.close()