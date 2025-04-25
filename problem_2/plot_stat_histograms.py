# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np # Import numpy for checking numeric types

# --- Configuration ---
CONFIG_FILE = 'config.json'
# Define the specific statistics to plot based on user request and config file names
STATS_TO_PLOT = [
    'Performance: goals',
    'Performance: assists',
    'Standard: shoots on target percentage (SoT%)',
    'Blocks: Int',
    'Performance: Recov',
    'Challenges: Att'
]

# --- Helper Functions ---

def load_config(config_path):
    """Loads the configuration from a JSON file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Successfully loaded configuration from {config_path}.")
        return config
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_path}'.")
        return None
    except Exception as e:
        print(f"An error occurred loading config: {e}")
        return None

def load_data(csv_path):
    """Loads player data from a CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: Data file '{csv_path}' not found. Please ensure it's uploaded and accessible.")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}.")
        # Basic check for required columns
        if 'Team' not in df.columns:
             print("Warning: 'Team' column not found. Team-specific plots cannot be generated.")
        return df
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_path} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred loading data: {e}")
        return None

def preprocess_data(df, stats_list):
    """Converts specified stat columns to numeric, handling errors."""
    if df is None:
        return None
    df_processed = df.copy()
    print("\n--- Preprocessing Data ---")
    for stat in stats_list:
        if stat in df_processed.columns:
            # Store original dtype
            original_dtype = df_processed[stat].dtype
            # Attempt conversion to numeric, coercing errors to NaN
            df_processed[stat] = pd.to_numeric(df_processed[stat], errors='coerce')
            # Check if conversion was successful (at least some values are numeric)
            if df_processed[stat].isnull().all() and not df[stat].isnull().all():
                 print(f"Warning: Column '{stat}' could not be converted to numeric and is now all NaN. Original dtype: {original_dtype}.")
            elif not pd.api.types.is_numeric_dtype(df_processed[stat]):
                 print(f"Warning: Column '{stat}' is not numeric after attempted conversion. Original dtype: {original_dtype}")
            # else:
            #      print(f"Column '{stat}' successfully processed as numeric.") # Optional: success message
        else:
            print(f"Warning: Statistic column '{stat}' not found in the DataFrame.")
    return df_processed


def plot_specific_histograms(df, stats_list):
    """Plots overall and per-team histograms for the specified list of statistics."""
    if df is None or df.empty:
        print("Cannot plot histograms because data is missing or empty.")
        return
    if 'Team' not in df.columns:
        print("Warning: 'Team' column missing. Skipping team-specific plots.")
        plot_team_hist = False
    else:
        plot_team_hist = True


    print("\n--- Plotting Histograms ---")
    for stat in stats_list:
        if stat not in df.columns:
            print(f"\nSkipping '{stat}': Column not found in data.")
            continue

        # Check if the column is numeric before plotting
        if not pd.api.types.is_numeric_dtype(df[stat]):
             print(f"\nSkipping '{stat}': Column is not numeric.")
             continue

        # Drop rows where the current stat is NaN for plotting
        stat_data = df.dropna(subset=[stat])

        if stat_data.empty:
            print(f"\nSkipping '{stat}': No valid data after dropping NaNs.")
            continue

        print(f"\nPlotting '{stat}'...")

        # 1. Overall Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(stat_data[stat], kde=True)
        plt.title(f'Distribution of {stat} (All Players)')
        plt.xlabel(stat)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # 2. Per-Team Histograms (only if 'Team' column exists and data is available)
        if plot_team_hist and 'Team' in stat_data.columns:
             teams = stat_data['Team'].unique()
             if len(teams) > 0:
                 # Use FacetGrid for a potentially large number of teams
                 try:
                    g = sns.FacetGrid(stat_data, col="Team", col_wrap=4, sharex=False, sharey=False, height=3, aspect=1.3)
                    g.map(sns.histplot, stat, kde=False) # kde=False can be cleaner for many small plots
                    g.fig.suptitle(f'Distribution of {stat} by Team', y=1.03) # Adjust title position
                    g.set_titles("{col_name}") # Set individual plot titles to team names
                    g.set_axis_labels(stat, "Frequency")
                    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout
                    plt.show()
                 except Exception as e:
                      print(f"  Could not generate team plots for '{stat}' using FacetGrid. Error: {e}")
                      # Fallback or skip if FacetGrid fails
             else:
                  print(f"  No teams found for statistic '{stat}' after filtering.")
        elif not plot_team_hist:
             print(f"  Skipping team plots for '{stat}' as 'Team' column is missing.")


# --- Main Execution ---
# Load configuration
config = load_config(CONFIG_FILE)

if config and 'csv_filename' in config:
    results_csv_path = config['csv_filename']
    # Load data
    df_players_raw = load_data(results_csv_path)

    if df_players_raw is not None:
        # Preprocess data (convert specified columns to numeric)
        df_players = preprocess_data(df_players_raw, STATS_TO_PLOT)

        # Plot the histograms for the specified statistics
        plot_specific_histograms(df_players, STATS_TO_PLOT)
    else:
        print("Could not load player data. Cannot generate plots.")
else:
    print("Could not load configuration or 'csv_filename' key is missing in config.")

