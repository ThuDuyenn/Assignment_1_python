# main.py
import os
import re
import time
import traceback
import logging
import json
from contextlib import contextmanager
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# --- Configuration Loading ---

def load_config(config_path='config.json'):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Construct paths for files *within* the output folder
        output_folder = config_data['output_folder']
        config_data['transfer_value_output_file'] = os.path.join(output_folder, config_data['transfer_value_output_filename'])
        config_data['estimation_ready_data_file'] = os.path.join(output_folder, config_data['estimation_ready_data_filename'])

        # Use the path directly from JSON for part1 results
        # Store it under a consistent key like 'part1_results_file'
        config_data['part1_results_file'] = config_data['part1_results_filename'] # <--- MODIFIED LOGIC

        return config_data
    except FileNotFoundError:
        logging.error(f"Configuration file not found at '{config_path}'")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from '{config_path}'")
        return None
    except KeyError as e:
        logging.error(f"Missing key in configuration file '{config_path}': {e}")
        return None

CONFIG = load_config()

# Exit if config loading failed
if CONFIG is None:
    logging.critical("Failed to load configuration. Exiting.")
    exit()


# Logging Setup (remains the same)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
selenium_logger = logging.getLogger('selenium.webdriver.remote.remote_connection')
selenium_logger.setLevel(logging.WARNING)


# --- Helper Functions (remain the same) ---

@contextmanager
def managed_webdriver(options: ChromeOptions):
    # ... (no changes needed here) ...
    driver = None
    try:
        logging.info("Initializing WebDriver...")
        driver = webdriver.Chrome(options=options)
        yield driver
        logging.info("WebDriver Context Yielded.")
    except Exception as e:
        logging.error(f"Failed to initialize or yield WebDriver: {e}")
        traceback.print_exc()
        raise
    finally:
        if driver:
            try:
                driver.quit()
                logging.info("WebDriver closed.")
            except Exception as e:
                logging.error(f"Error closing WebDriver: {e}")

def safe_get_text(element, selector: str, default='N/a') -> str:
    # ... (no changes needed here) ...
    try:
        target = element.select_one(selector)
        return target.get_text(strip=True) if target else default
    except Exception as e:
        logging.warning(f"Error extracting text with selector '{selector}': {e}")
        return default


def clean_transfer_value(value_str: str, config: dict) -> float | None:
    # ... (no changes needed here) ...
    target_variable = config['processing']['target_variable'] # Example of using config

    if not isinstance(value_str, str) or value_str == 'N/a':
        return None
    value_str = re.sub(r'[€£$]', '', value_str.lower().strip())
    multiplier = 1.0
    if 'm' in value_str:
        multiplier = 1_000_000
        value_str = value_str.replace('m', '').strip()
    elif 'k' in value_str:
        multiplier = 1_000
        value_str = value_str.replace('k', '').strip()
    try:
        return (float(value_str) * multiplier) / 1_000_000
    except ValueError:
        logging.debug(f"Could not convert value: {value_str}")
        return None

# --- Core Logic (remain the same, will use updated CONFIG['part1_results_file']) ---

def scrape_transfer_data(driver: WebDriver, config: dict) -> dict:
    # ... (no changes needed here) ...
    player_data = {}
    page_num = 1
    # Access config values using dictionary keys
    wait = WebDriverWait(driver, config['scraping']['wait_time'])
    transfer_url = config['scraping']['transfer_url']
    player_table_selector = config['scraping']['player_table_selector']
    player_row_selector = config['scraping']['player_row_selector']
    player_name_selector = config['scraping']['player_name_selector']
    team_name_selector = config['scraping']['team_name_selector']
    etv_selector = config['scraping']['etv_selector']
    next_page_selector = config['scraping']['next_page_selector']
    target_variable = config['processing']['target_variable']

    logging.info(f"Starting transfer value scraping from: {transfer_url}")
    try:
        driver.get(transfer_url)
    except Exception as e:
        logging.error(f"Error navigating to initial URL {transfer_url}: {e}")
        return {}

    while True:
        logging.info(f"--- Scraping Page {page_num} ---")
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"{player_table_selector} {player_row_selector}")))
            time.sleep(1)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            player_table = soup.select_one(player_table_selector)
            if not player_table:
                logging.warning(f"Player table '{player_table_selector}' not found on page {page_num}. Stopping.")
                break

            player_rows = player_table.select(player_row_selector)
            logging.info(f"Found {len(player_rows)} player rows on page {page_num}.")
            rows_processed_this_page = 0

            for row in player_rows:
                player = safe_get_text(row, player_name_selector)
                team = safe_get_text(row, team_name_selector)
                etv = safe_get_text(row, etv_selector)

                if player == 'N/a' or team == 'N/a':
                    continue

                player_key = (player, team)
                if player_key not in player_data:
                    player_data[player_key] = {
                        'Player': player,
                        'Team_TransferSite': team,
                        'TransferValueRaw': etv,
                        target_variable: clean_transfer_value(etv, config) # Pass config if needed
                    }
                    rows_processed_this_page += 1

            logging.info(f"Processed {rows_processed_this_page} new entries from page {page_num}.")

            # Pagination
            try:
                next_page_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, next_page_selector)))
                logging.info("Clicking 'Next page' button...")
                driver.execute_script("arguments[0].click();", next_page_button)
                time.sleep(2)
                page_num += 1
            except (NoSuchElementException, TimeoutException):
                logging.info("No more clickable 'Next page' button found. Ending scraping.")
                break

        except TimeoutException:
             logging.warning(f"Timeout waiting for table elements on page {page_num}. Stopping.")
             break
        except Exception as e:
            logging.error(f"Unexpected error during scraping on page {page_num}: {e}")
            traceback.print_exc()
            break

    logging.info(f"Finished scraping. Total unique (Player, Team) combinations found: {len(player_data)}")
    return player_data


def process_data(scraped_data: dict, config: dict) -> pd.DataFrame | None:
    # ... (no changes needed here, it uses config['part1_results_file']) ...
    # Access config values using dictionary keys
    part1_results_file = config['part1_results_file'] # Uses the path set in load_config
    minutes_col = config['processing']['part1_minutes_column']
    min_minutes = config['processing']['min_minutes_threshold']
    player_col = config['processing']['part1_player_column']
    team_col = config['processing']['part1_team_column']
    target_variable = config['processing']['target_variable']
    output_folder = config['output_folder']
    transfer_output_file = config['transfer_value_output_file']
    estimation_output_file = config['estimation_ready_data_file']

    # 1. Load Part 1 Data
    try:
        # This line now correctly uses the relative path
        df_part1 = pd.read_csv(part1_results_file)
        logging.info(f"Loaded Part 1 data: {df_part1.shape[0]} players from '{part1_results_file}'.")
    except FileNotFoundError:
        logging.error(f"Part 1 results file not found at '{part1_results_file}'")
        return None
    except Exception as e:
        logging.error(f"Error loading Part 1 data from '{part1_results_file}': {e}")
        return None

    # 2. Filter Part 1 by Minutes
    if minutes_col not in df_part1.columns:
        logging.error(f"Minutes column '{minutes_col}' not found.")
        return None

    df_part1['minutes_numeric'] = pd.to_numeric(
        df_part1[minutes_col].astype(str).str.replace(',', '', regex=False), errors='coerce'
    )
    df_filtered = df_part1.dropna(subset=['minutes_numeric'])
    df_filtered = df_filtered[df_filtered['minutes_numeric'] > min_minutes].copy()
    df_filtered.drop(columns=['minutes_numeric'], inplace=True)

    if df_filtered.empty:
        logging.warning(f"No players found with > {min_minutes} minutes.")
        return df_filtered

    logging.info(f"Filtered Part 1 data to {df_filtered.shape[0]} players.")

    # 3. Prepare Transfer Data DataFrame
    if not scraped_data:
        logging.warning("No scraped transfer data provided. Adding empty transfer columns.")
        df_merged = df_filtered.copy()
        df_merged['Team_TransferSite'] = 'N/a'
        df_merged['TransferValueRaw'] = 'N/a'
        df_merged[target_variable] = pd.NA
    else:
        try:
            df_transfer = pd.DataFrame.from_dict(scraped_data, orient='index').reset_index()
            if 'level_0' in df_transfer.columns and 'level_1' in df_transfer.columns:
                 df_transfer.rename(columns={'level_0': 'Player', 'level_1': 'Team_TransferSite'}, inplace=True)
            elif 'index' in df_transfer.columns and isinstance(df_transfer['index'].iloc[0], tuple):
                 df_transfer[['Player', 'Team_TransferSite']] = pd.DataFrame(df_transfer['index'].tolist(), index=df_transfer.index)
                 df_transfer.drop(columns=['index'], inplace=True)

            logging.info(f"Converted scraped data to DataFrame: {df_transfer.shape[0]} entries.")

            # 4. Merge Data
            if not all(col in df_filtered.columns for col in [player_col, team_col]):
                 logging.error("Missing Player/Team columns in filtered Part 1 data for merge.")
                 return df_filtered

            if not all(col in df_transfer.columns for col in ['Player', 'Team_TransferSite', target_variable]):
                logging.error(f"Missing required columns in transfer DataFrame for merge. Found: {df_transfer.columns.tolist()}")
                return df_filtered

            logging.info(f"Merging on Part1:({player_col}, {team_col}) <-> Transfer:('Player', 'Team_TransferSite')")
            df_merged = pd.merge(
                df_filtered,
                df_transfer[['Player', 'Team_TransferSite', 'TransferValueRaw', target_variable]],
                left_on=[player_col, team_col],
                right_on=['Player', 'Team_TransferSite'],
                how='left'
            )
            df_merged[target_variable].fillna(pd.NA, inplace=True)
            df_merged['TransferValueRaw'].fillna('N/a', inplace=True)
            if 'Team_TransferSite' in df_merged.columns:
                 df_merged['Team_TransferSite'].fillna('N/a', inplace=True)
            else:
                 df_merged['Team_TransferSite'] = 'N/a'

            logging.info(f"Merged data shape: {df_merged.shape}. Missing transfer values: {df_merged[target_variable].isna().sum()}")

        except Exception as e:
            logging.error(f"Error during transfer data processing or merge: {e}")
            traceback.print_exc()
            df_merged = df_filtered.copy()
            df_merged['Team_TransferSite'] = 'N/a'
            df_merged['TransferValueRaw'] = 'N/a'
            df_merged[target_variable] = pd.NA

    # 5. Save Results
    try:
        os.makedirs(output_folder, exist_ok=True)
        df_merged.to_csv(transfer_output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Saved merged data to '{transfer_output_file}'")
        df_merged.to_csv(estimation_output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Saved estimation-ready data to '{estimation_output_file}'")
        return df_merged
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        return df_merged


# --- Main Execution (remains the same) ---

def main():
    """Main function to run the scraping and processing pipeline."""
    start_time = time.time()
    logging.info("--- Start Part IV: Transfer Value Collection & Processing ---")

    # Check if config was loaded successfully
    if CONFIG is None:
        return

    chrome_options = ChromeOptions()
    chrome_options.add_argument('--log-level=3')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    final_df = None
    try:
        with managed_webdriver(chrome_options) as driver:
            scraped_data = scrape_transfer_data(driver, CONFIG)
            final_df = process_data(scraped_data, CONFIG)

        if final_df is None:
             logging.warning("Processing did not return a DataFrame.")
        elif final_df.empty:
             logging.info("Processing resulted in an empty DataFrame.")
        else:
             logging.info(f"Successfully processed data. Final shape: {final_df.shape}")

    except Exception as e:
        logging.critical(f"An critical error occurred during the main execution: {e}")
        traceback.print_exc()

    finally:
        end_time = time.time()
        logging.info(f"--- Part IV finished in {end_time - start_time:.2f} seconds ---")


if __name__ == '__main__':
    main()