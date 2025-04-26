import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'problem_1', 'results.csv')
output_filename = 'results2.csv'
output_path = os.path.join(current_dir, output_filename)

na_values_list = ['N/a', 'n/a', 'NA', 'na', 'NaN', 'nan', '']
df = pd.read_csv(csv_path, na_values=na_values_list)
 
exclude_cols = [
    'Name', 'Nation', 'Team', 'Position', 'Age'
]

stats_columns = [col for col in df.columns if col not in exclude_cols]

overall_stats = df[stats_columns].agg(['median', 'mean', 'std']).T 
overall_stats.columns = ['Median', 'Mean', 'Std'] 

all_row = pd.DataFrame(index=['all'])

for col in stats_columns:
    all_row[f'Median of {col}'] = overall_stats.loc[col, 'Median']
    all_row[f'Mean of {col}'] = overall_stats.loc[col, 'Mean']
    all_row[f'Std of {col}'] = overall_stats.loc[col, 'Std']

team_stats = df.groupby('Team')[stats_columns].agg(['median', 'mean', 'std'])

team_results = pd.DataFrame(index=team_stats.index)

for col in stats_columns:
    team_results[f'Median of {col}'] = team_stats[(col, 'median')] 
    team_results[f'Mean of {col}'] = team_stats[(col, 'mean')]
    team_results[f'Std of {col}'] = team_stats[(col, 'std')]

all_row.index.name = 'Identifier' 
all_row = all_row.reset_index()
all_row = all_row.rename(columns={'Identifier': ''}) 

team_results.index.name = '' 
team_results = team_results.reset_index()

final_results = pd.concat([all_row, team_results], ignore_index=True)

final_results.to_csv(output_path, index=False, encoding='utf-8')
print(final_results.head())
