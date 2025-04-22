'''
Note nhung van de can xu ly:
    - loc nhung cau thu thi dau tren 90 phut
    - sap xep theo first name
    - co truong hop mot cau thu da cho nhieu hon 1 team trong 1 mua giai vi cac ly do nhu cho muon or ki chuyen nhuong mua dong:
    VD: Trevoh Chalobah cua Chelsea cho Crystal Palace muon nhung giua mua lai duoc goi ve da
    - lam sach du lieu:
        + format lai nhung data can thiet
        + khong co data -> N/a
'''

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import re
import json
import sys

# --- Configuration Loading ---
CONFIG_FILE = 'config.json'

def load_config(filename):
    # try:
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Configuration loaded successfully from {filename}.")
        # Basic validation (check if essential keys exist)
        required_keys = ["base_url", "season_url_suffix", "csv_filename",
                         "min_minutes", "wait_timeout", "stats_config", "column_order"]
        # for key in required_keys:
        #     if key not in config:
        #         raise ValueError(f"Missing required key '{key}' in {filename}")
        return config
    # except FileNotFoundError:
    #     print(f"ERROR: Configuration file '{filename}' not found.")
    #     sys.exit(1) # Exit if config file is missing
    # except json.JSONDecodeError:
    #     print(f"ERROR: Could not decode JSON from '{filename}'. Check its format.")
    #     sys.exit(1) # Exit if JSON is invalid
    # except ValueError as ve:
    #     print(f"ERROR: Invalid configuration: {ve}")
    #     sys.exit(1) # Exit on validation error
    # except Exception as e:
    #     print(f"ERROR: An unexpected error occurred loading configuration: {e}")
    #     sys.exit(1) # Exit on other loading errors

config = load_config(CONFIG_FILE)

BASE_URL = config["base_url"]
SEASON_URL_SUFFIX = config["season_url_suffix"]
STATS_CONFIG = config["stats_config"] 
COLUMN_ORDER = config["column_order"] 
CSV_FILENAME = config["csv_filename"]
MIN_MINUTES = config["min_minutes"]
WAIT_TIMEOUT = config["wait_timeout"]

def safe_get_text(element, data_stat, default="N/a"):
    # """ Safely extracts text from a child td element with a specific data-stat. """
    try:
        cell = element.find('td', attrs={'data-stat': data_stat})
        # Strip text and return "N/a" if the result is empty after stripping
        text = cell.get_text(strip=True) if cell else ''
        return text if text else default
    except Exception:
        return default

def get_player_id(player_cell):
    # """ Extracts the unique player ID from the player cell's link. """
    try:
        link = player_cell.find('a')
        if link and link.has_attr('href'):
            # Example href: /en/players/1e210d97/Granit-Xhaka -> extract '1e210d97'
            match = re.search(r'/players/([a-zA-Z0-9]+)/', link['href'])
            if match:
                return match.group(1)
    except Exception:
        pass
    return None

def format_value(stat_name, value):
    """ Formats specific values (Nation, Age) and handles empty strings or None. """
    # Ensure the default "N/a" is returned for None or empty strings
    if value is None or value == '':
        return "N/a"

    # Apply specific formatting only if the value is not already "N/a"
    if value != "N/a":
        if stat_name == 'Nation':
            # Extracts the 2 or 3 letter code (e.g., 'ENG' from 'ENG England')
            return value.split()[0] if value and ' ' in value else value # Handle cases with only code
        elif stat_name == 'Age':
            # Extracts the age number (e.g., '25' from '25-123')
            return value.split('-')[0] if '-' in value else value
        # Remove commas from numbers for potential conversion later
        elif isinstance(value, str) and ',' in value:
             # Check if it's likely a number before removing comma
             if re.match(r'^-?[\d,]+(\.\d+)?$', value):
                 return value.replace(',', '')

    # Return the original value if no specific formatting applied or if it was already "N/a"
    return value

# --- Main Scraping Function ---

def scrape_fbref_data(driver, url, table_id, stats_to_extract, player_data, global_stats_config):
    """
    Scrapes a single table from FBRef and updates the player_data dictionary.

    Args:
        driver: Selenium WebDriver instance.
        url (str): The URL of the page to scrape.
        table_id (str): The HTML ID of the table to scrape.
        stats_to_extract (list): List of 'data-stat' attributes to extract from this table.
        player_data (dict): The main dictionary holding all player stats, keyed by player_id.
        global_stats_config (dict): The loaded STATS_CONFIG for mapping.
    """
    print(f"Scraping table '{table_id}' from {url}...")
    try:
        driver.get(url)
        wait = WebDriverWait(driver, WAIT_TIMEOUT)
        # Wait for the table body to ensure data is loaded
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"table#{table_id} tbody tr")))
        time.sleep(1) # Small delay for safety

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        table = soup.find('table', id=table_id)

        if not table:
            print(f"  WARNING: Table '{table_id}' not found on {url}.")
            return

        rows = table.select('tbody tr:not(.thead, .spacer_league)') # Select data rows, exclude headers/spacers

        if not rows:
            print(f"  WARNING: No data rows found in table '{table_id}' on {url}.")
            return

        processed_count = 0
        for row in rows:
            player_cell = row.find('td', attrs={'data-stat': 'player'})
            if not player_cell:
                continue

            player_id = get_player_id(player_cell)
            player_name = player_cell.get_text(strip=True) # Get name even if ID fails

            player_key = player_id if player_id else player_name
            if not player_key:
                    continue

            if player_key not in player_data:
                player_data[player_key] = {'player_id': player_id, 'Name': player_name}

            for stat in stats_to_extract:
                # Find the corresponding CSV column name using the loaded config
                stat_config_key = next((k for k, v in global_stats_config.items() if v['data_stat'] == stat and v['table_id'] == table_id), None)
                if stat_config_key:
                        value = safe_get_text(row, stat, default="N/a")
                        player_data[player_key][stat_config_key] = value

            processed_count += 1
        print(f"  Processed {processed_count} player rows from table '{table_id}'.")

    except TimeoutException:
        print(f"  ERROR: Timed out waiting for table '{table_id}' on {url}.")
    except NoSuchElementException:
        print(f"  ERROR: Could not find expected elements for table '{table_id}' on {url}.")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred scraping {url} table '{table_id}': {e}")

# --- Main Execution ---

if __name__ == "__main__":
    start_time = time.time()
    all_players_data = {}

    # Group configurations by URL using the loaded STATS_CONFIG
    configs_by_url = {}
    for csv_col, cfg in STATS_CONFIG.items(): # Use loaded STATS_CONFIG
        full_url = BASE_URL + cfg['url_suffix'] + SEASON_URL_SUFFIX
        table_id = cfg['table_id']
        data_stat = cfg['data_stat']

        if full_url not in configs_by_url:
            configs_by_url[full_url] = {}
        if table_id not in configs_by_url[full_url]:
            configs_by_url[full_url][table_id] = set()

        configs_by_url[full_url][table_id].add(data_stat)

    # Initialize WebDriver
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = None

    try:
        print("Initializing WebDriver...")
        driver = webdriver.Chrome(options=options)
        print("WebDriver initialized.")

            # Scrape data table by table, URL by URL
        for url, tables in configs_by_url.items():
            for table_id, stats_set in tables.items():
                # Pass the loaded STATS_CONFIG for mapping inside the function
                scrape_fbref_data(driver, url, table_id, list(stats_set), all_players_data, STATS_CONFIG)
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred during the scraping process: {e}")
    finally:
        if driver:
            print("Closing WebDriver.")
            driver.quit()

    print(f"\nTotal players scraped initially: {len(all_players_data)}")

    # --- Data Processing & Filtering ---
    final_data_list = []
    if all_players_data:
        print(f"Filtering players by minutes played > {MIN_MINUTES}...")
        filtered_players = 0
        players_to_process = list(all_players_data.values())

        for stats_dict in players_to_process:
            try:
                minutes_raw = stats_dict.get('Playing Time: minutes', '0')
                minutes_str = str(minutes_raw).replace(',', '') if minutes_raw is not None else '0'
                minutes_played = int(minutes_str) if minutes_str.isdigit() else 0
            except (ValueError, TypeError):
                minutes_played = 0

            if minutes_played > MIN_MINUTES:
                player_row = []
                # Build the row according to the loaded COLUMN_ORDER
                for col_name in COLUMN_ORDER: # Use loaded COLUMN_ORDER
                    raw_value = stats_dict.get(col_name, None)
                    formatted_val = format_value(col_name, raw_value)
                    player_row.append(formatted_val)
                final_data_list.append(player_row)
                filtered_players += 1
        print(f"Players meeting criteria: {filtered_players}")

        if final_data_list:
            print("Sorting players alphabetically by name...")
            final_data_list.sort(key=lambda x: x[0])
        else:
            print("No players met the filtering criteria.")

    else:
        print("No player data was collected.")

    # --- Save to CSV ---
    if final_data_list:
        print(f"Saving data for {len(final_data_list)} players to {CSV_FILENAME}...")
        try:
            # Create DataFrame using the loaded COLUMN_ORDER
            df = pd.DataFrame(final_data_list, columns=COLUMN_ORDER)
            df.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
            print("Data saved successfully.")
        except Exception as e:
            print(f"ERROR: Failed to save data to CSV: {e}")
    else:
        print("No data to save.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

