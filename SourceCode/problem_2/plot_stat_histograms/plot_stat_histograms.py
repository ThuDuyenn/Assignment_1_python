import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math

# --- Config ---
TEAMS_PER_PLOT = 20
TEAM_COL = 'Team'
PLOT_STYLE = "whitegrid"
MAX_FD_BINS = 100 

script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder_name = 'stat_histograms'
OUTPUT_DIR = os.path.join(script_dir, output_folder_name)

# --- Helpers ---
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_filename(name: str, max_len: int = 100) -> str:
    sanitized = "".join(c if c.isalnum() else "_" for c in name)
    sanitized = "_".join(filter(None, sanitized.split('_')))
    return sanitized.strip('_')[:max_len]

def calculate_fd_bins(data: pd.Series) -> int:
    """Tính toán số lượng bins tối ưu sử dụng quy tắc Freedman-Diaconis."""
    data = data.dropna()
    n = len(data)
    if n < 2: return 1
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    data_min, data_max = data.min(), data.max()
    if data_max == data_min: return 1

    num_bins = 0
    if iqr > 0:
        bin_width = 2 * iqr / np.cbrt(n)
        if bin_width > 0:
            num_bins = math.ceil((data_max - data_min) / bin_width)
    if num_bins <= 0: # Fallback (Sturges' Rule)
        num_bins = math.ceil(1 + np.log2(n))

    num_bins = max(1, num_bins)
    if MAX_FD_BINS > 0: num_bins = min(num_bins, MAX_FD_BINS)
    return int(num_bins)

# --- Plotting Functions ---
def plot_overall_hist(df: pd.DataFrame, stat: str, output_dir: str):
    ensure_dir(output_dir)
    if stat not in df.columns: return

    numeric_data = pd.to_numeric(df[stat], errors='coerce').dropna()
    if numeric_data.empty: return

    bins = calculate_fd_bins(numeric_data)
    if bins <= 0: bins = 10 

    plt.figure(figsize=(10, 6))
    sns.histplot(numeric_data, bins=bins, kde=True)
    plt.title(f'Overall Distribution: {stat} (Bins: {bins})')
    plt.xlabel(stat); plt.ylabel('Frequency')
    plt.ylim(bottom=0)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"hist_overall_{sanitize_filename(stat)}.png")
    plt.savefig(filepath)
    plt.close()

def plot_team_hist(df: pd.DataFrame, stat: str, output_dir: str, team_col: str = TEAM_COL, teams_per_fig: int = TEAMS_PER_PLOT):
    ensure_dir(output_dir)
    if stat not in df.columns or team_col not in df.columns: return

    df_copy = df[[team_col, stat]].copy()
    df_copy[stat] = pd.to_numeric(df_copy[stat], errors='coerce')
    if df_copy[stat].isnull().all(): return

    teams = sorted(df_copy[team_col].dropna().unique())
    num_teams = len(teams)
    if num_teams == 0: return

    for i in range(math.ceil(num_teams / teams_per_fig)):
        start_idx, end_idx = i * teams_per_fig, (i + 1) * teams_per_fig
        current_teams = teams[start_idx:end_idx]
        fig_data = df_copy[df_copy[team_col].isin(current_teams)].dropna(subset=[stat])
        if fig_data.empty or fig_data[stat].empty: continue

        bins = calculate_fd_bins(fig_data[stat])
        if bins <= 0: bins = 10 

        ncols = min(5, len(current_teams))
        nrows = math.ceil(len(current_teams) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)
        axes = axes.flatten()

        x_min, x_max = fig_data[stat].min(), fig_data[stat].max()
        if x_min == x_max: x_min -= 0.5; x_max += 0.5
        elif (x_max - x_min) < bins * 1e-9: 
             padding = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 0.5
             x_min -= padding
             x_max += padding

        for j, team in enumerate(current_teams):
            ax = axes[j]
            team_data = fig_data[fig_data[team_col] == team][stat]
            if not team_data.empty:
                sns.histplot(team_data, bins=bins, kde=True, ax=ax, binrange=(x_min, x_max))
                ax.set_title(team, fontsize=9)
                ax.set_xlabel(''); ax.set_ylabel('')
                ax.tick_params(labelsize=8)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(bottom=0)
            else:
                ax.set_visible(False)

        for k in range(len(current_teams), len(axes)): axes[k].set_visible(False)

        fig.suptitle(f'{stat} Distribution by Team (Group {i+1}, Bins: {bins})', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        filename = f"hist_teams_{sanitize_filename(stat)}_group_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)

# --- Main Execution ---
if __name__ == '__main__':
    sns.set_theme(style=PLOT_STYLE)
    output_final_dir = OUTPUT_DIR 

    stats_to_plot = [
        'Performance: goals',
        'Performance: assists',
        'Standard: shoots on target percentage (SoT%)',
        'Blocks: Int',
        'Performance: Recov',
        'Challenges: Att'
    ]

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_script_dir, '..', '..', 'problem_1', 'results1.csv')
    df = pd.read_csv(csv_path, encoding='utf-8')
        
    ensure_dir(output_final_dir)

    for stat in stats_to_plot:
        plot_overall_hist(df, stat, output_dir=output_final_dir)
        plot_team_hist(df, stat, output_dir=output_final_dir)