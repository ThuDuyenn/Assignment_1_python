import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', '..', 'problem_1', 'results1.csv')
output_filename = 'top_3_formatted.txt'
output_path = os.path.join(current_dir, output_filename)
df = pd.read_csv(csv_path, na_values=['N/a']) 

exclude_cols = [
    'Name', 'Nation', 'Team', 'Position'
]

stats_columns = [col for col in df.columns if col not in exclude_cols]

with open(output_path, 'w', encoding='utf-8') as f:
    for col in stats_columns:
        f.write(f"--- {col} ---\n")
        f.write("\n")

        original_dtype = df[col].dtype
        numeric_col_series = pd.to_numeric(df[col], errors='coerce')
        can_sort_numeric = False
        df_for_sort = pd.DataFrame()

        if numeric_col_series.notna().any() and pd.api.types.is_numeric_dtype(numeric_col_series):
            df_for_sort = df[['Name', 'Team']].copy()
            df_for_sort[col] = numeric_col_series
            df_for_sort.dropna(subset=[col], inplace=True)
            if not df_for_sort.empty:
                can_sort_numeric = True

        if can_sort_numeric:
            top_3 = df_for_sort.sort_values(col, ascending=False).head(3)
            f.write("Top 3 Players:\n")
            if all(c in top_3.columns for c in ['Name', 'Team', col]):
                 f.write(top_3[['Name', 'Team', col]].to_string(index=False, header=True))
            f.write("\n\n")

            bottom_3 = df_for_sort.sort_values(col, ascending=True).head(3)
            f.write("Bottom 3 Players:\n")
            if all(c in bottom_3.columns for c in ['Name', 'Team', col]):
                 f.write(bottom_3[['Name', 'Team', col]].to_string(index=False, header=True))
            f.write("\n\n")

        f.write("=" * 50 + "\n\n")