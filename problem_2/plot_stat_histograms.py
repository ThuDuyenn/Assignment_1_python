import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math
from typing import List

# --- Config ---
OUTPUT_DIR = 'stat_histograms_real_data_relative' 
TEAMS_PER_PLOT = 20
DEFAULT_BINS = 15
TEAM_COL = 'Team' 
PLOT_STYLE = "whitegrid"

# --- Helpers ---
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_filename(name: str, max_len: int = 100) -> str:
    sanitized = "".join(c if c.isalnum() else "_" for c in name)
    sanitized = "_".join(filter(None, sanitized.split('_')))
    return sanitized.strip('_')[:max_len]

# --- Plotting Functions ---
def plot_overall_hist(df: pd.DataFrame, stat: str, **kwargs):
    output_dir = kwargs.get('output_dir', OUTPUT_DIR)
    bins = kwargs.get('bins', DEFAULT_BINS)
    ensure_dir(output_dir)

    if stat not in df.columns or df[stat].dropna().empty:
        return

    plt.figure(figsize=(10, 6))
    numeric_data = pd.to_numeric(df[stat], errors='coerce').dropna()
    if numeric_data.empty:
        return

    sns.histplot(numeric_data, bins=bins, kde=True)
    plt.title(f'Overall Distribution: {stat}')
    plt.xlabel(stat); plt.ylabel('Frequency')
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"hist_overall_{sanitize_filename(stat)}.png")
    plt.savefig(filepath)
    plt.close()

def plot_team_hist(df: pd.DataFrame, stat: str, **kwargs):
    output_dir = kwargs.get('output_dir', OUTPUT_DIR)
    bins = kwargs.get('bins', DEFAULT_BINS)
    teams_per_fig = kwargs.get('teams_per_fig', TEAMS_PER_PLOT)
    team_col = kwargs.get('team_col', TEAM_COL)
    ensure_dir(output_dir)

    if stat not in df.columns or team_col not in df.columns:
        return

    df_copy = df[[team_col, stat]].copy()
    df_copy[stat] = pd.to_numeric(df_copy[stat], errors='coerce')

    if df_copy[stat].isnull().all():
        return

    teams = sorted(df_copy[team_col].dropna().unique())
    num_teams = len(teams)
    if num_teams == 0: return

    for i in range(math.ceil(num_teams / teams_per_fig)):
        start, end = i * teams_per_fig, (i + 1) * teams_per_fig
        current_teams = teams[start:end]
        fig_data = df_copy[df_copy[team_col].isin(current_teams)].dropna(subset=[stat])

        if fig_data.empty: continue

        ncols = min(5, len(current_teams))
        nrows = math.ceil(len(current_teams) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)
        axes = axes.flatten()

        x_min, x_max = fig_data[stat].min(), fig_data[stat].max()
        if x_min == x_max: x_min -= 0.5; x_max += 0.5

        for j, team in enumerate(current_teams):
            ax = axes[j]
            team_data = fig_data[fig_data[team_col] == team][stat]
            if not team_data.empty:
                sns.histplot(team_data, bins=bins, kde=False, ax=ax)
                ax.set_title(team, fontsize=9)
                ax.set_xlabel(''); ax.set_ylabel('')
                try: ax.set_xlim(x_min, x_max)
                except ValueError: pass
            else:
                 ax.set_visible(False)

        for k in range(len(current_teams), len(axes)): axes[k].set_visible(False)

        fig.suptitle(f'{stat} Distribution by Team (Group {i+1})', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        filepath = os.path.join(output_dir, f"hist_teams_{sanitize_filename(stat)}_group_{i+1}.png")
        
        plt.savefig(filepath)

        plt.close(fig)

# --- Main Execution ---
if __name__ == '__main__':
    sns.set_theme(style=PLOT_STYLE)

    stats_to_plot = [
        'Performance: goals',
        'Performance: assists',
        'Standard: shoots on target percentage (SoT%)',
        'Blocks: Int',
        'Performance: Recov',
        'Challenges: Att'
    ]

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_script_dir, '..', 'problem_1', 'results.csv')
    df = pd.read_csv(csv_path, encoding='utf-8')
        
    for stat in stats_to_plot:
        plot_overall_hist(df, stat)

    for stat in stats_to_plot:
            plot_team_hist(df, stat)

    print(f"\nDone. Check plots in '{OUTPUT_DIR}'.")