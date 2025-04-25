import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'problem_1', 'results.csv')
output_filename = 'top_3_formatted.txt'
output_path = os.path.join(current_dir, output_filename)
df = pd.read_csv(csv_path, na_values=['N/a']) 

exclude_cols = [
    'Name', 'Nation', 'Team', 'Position', 'Age',
    'Playing Time: matches played', 'Playing Time: starts'
]

stats_columns = [col for col in df.columns if col not in exclude_cols]

with open(output_path, 'w', encoding='utf-8') as f:
    for col in stats_columns:
        f.write(f"--- {col} ---\n") 
        f.write("-" * (len(col) + 6) + "\n\n")

        top_3 = df.dropna(subset=[col]).sort_values(col, ascending=False)[['Name', col]].head(3)
        f.write("Top 3 Player:\n")
    
        f.write(top_3.to_string(index=False, header=True))
        
        f.write("\n\n")

        bottom_3 = df.dropna(subset=[col]).sort_values(col, ascending=True)[['Name', col]].head(3)
        f.write("Bottom 3 Player:\n")

        f.write(bottom_3.to_string(index=False, header=True))
        
        f.write("\n\n") 
        # --- Dấu phân cách Phần ---
        f.write("=" * 50 + "\n\n")