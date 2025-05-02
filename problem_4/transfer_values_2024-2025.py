import os
import re
import time
import traceback
import pandas as pd
import random
import json
from contextlib import contextmanager
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import ( NoSuchElementException, TimeoutException, StaleElementReferenceException, ElementClickInterceptedException )
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from typing import Optional, Dict, Any, Tuple, Set, Generator 

# --- Global Variables ---
CONFIG: Optional[Dict[str, Any]] = None 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# --- Configuration Loading ---
def load_config(config_path='config.json'):
    abs_config_path = os.path.abspath(os.path.join(SCRIPT_DIR, config_path))
    with open(abs_config_path, 'r') as f:
            config_data = json.load(f)
    return config_data
    
# --- WebDriver Context Manager ---
@contextmanager
def managed_driver(config: Dict[str, Any]) -> Generator[Optional[WebDriver], None, None]:
    driver = None
    options = ChromeOptions()
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f"user-agent={config['scraping']['user_agent']}")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(45)
    yield driver
    driver.quit()
    
# --- Helper Functions ---
def safe_get_text_transfer(element, selector):
    try:
        target = element.select_one(selector)
        return target.get_text(strip=True) if target else 'N/a'
    except Exception: return 'N/a'

def safe_get_numeric(text_value):
    if isinstance(text_value, str): text_value = text_value.replace(',', '')
    return float(text_value)

def clean_transfer_value(value_str):
    if value_str == 'N/a' or not isinstance(value_str, str): 
        return None
    value_str = re.sub(r'[€£$]', '', value_str.strip()).replace(',', '').lower()
    mult = 1_000_000.0 if 'm' in value_str else (1_000.0 if 'k' in value_str else 1.0)
    value_str = value_str.replace('m', '').replace('k', '').strip()
    return float(value_str) * mult / 1_000_000.0

def get_next_page_button(driver, config: Dict[str, Any]): # Now accepts config
    try:
        wait = WebDriverWait(driver, 5)
        selector = config['selectors']['next_page_enabled']
        next_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
        return next_btn
    except (NoSuchElementException, TimeoutException): return None
    except Exception as e: return None

def save_data(df: pd.DataFrame, full_path: str, output_folder: str) -> bool:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"Successfully saved {len(df)} records to '{full_path}'")
    return True

# --- Scraper Function ---
PlayerData = Dict[Tuple[str, str], Dict[str, Any]]

def scrape_transfer_data(driver: WebDriver, config: Dict[str, Any]) -> Optional[PlayerData]:
    player_transfer_values: PlayerData = {}
    page_num = 1
    max_page_retries = 2
    max_stale_retries = 3

    transfer_url = config['scraping']['transfer_url']
    wait_time = config['scraping']['wait_time_seconds']
    selectors = config['selectors']
    target_variable = config['processing']['target_variable_name']

    driver.get(transfer_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    wait = WebDriverWait(driver, wait_time)

    # --- Inside the scrape_transfer_data function ---
    while True:
        try:
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selectors['player_table'])))
            last_row_selector = f"{selectors['player_table']} {selectors['player_row']}:last-child"
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, last_row_selector)))
            time.sleep(0.75) 

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            table_body = soup.select_one(selectors['player_table'])
           
            if not table_body:
                print(f"Error: Player table body selector '{selectors['player_table']}' not found on page {page_num}. Stopping scrape.")
                break 

            rows = table_body.select(selectors['player_row'])
            
            if not rows:
                print(f"Warning: No player rows found using selector '{selectors['player_row']}' on page {page_num}. Checking for next page.")
               
            # --- Extract data from rows ---
            players_added_on_page = 0
            for row_index, row in enumerate(rows):
                player_name = safe_get_text_transfer(row, selectors['player_name'])
                team_name = safe_get_text_transfer(row, selectors['team_name'])
                
                if 'N/a' in (player_name, team_name) or not player_name or not team_name:
                    continue 

                key = (player_name.strip(), team_name.strip())
                
                if key not in player_transfer_values:
                    etv_str = safe_get_text_transfer(row, selectors['etv'])
                    player_transfer_values[key] = {
                        'Player': key[0], 
                        'Team_TransferSite': key[1],
                        'Age': safe_get_numeric(safe_get_text_transfer(row, selectors['age'])),
                        'Position': safe_get_text_transfer(row, selectors['position']),
                        'Skill': safe_get_numeric(safe_get_text_transfer(row, selectors['skill'])),
                        'Potential': safe_get_numeric(safe_get_text_transfer(row, selectors['potential'])),
                        'TransferValueRaw': etv_str,
                        target_variable: clean_transfer_value(etv_str)
                    }
                    players_added_on_page += 1
                    
            print(f"Successfully processed page {page_num}. Found {len(rows)} rows, added {players_added_on_page} new unique players.")

        # --- Unified Exception Handling for Page Processing ---
        except (TimeoutException, StaleElementReferenceException, NoSuchElementException, Exception) as e_page:
            return player_transfer_values 

        next_btn = get_next_page_button(driver, config)

        if not next_btn:
            break 

        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
            time.sleep(random.uniform(0.4, 0.8))

            try:
                next_btn.click()
            except ElementClickInterceptedException:
                print("Standard click intercepted, attempting JavaScript click for pagination.")
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(0.2) 

            print(f"Clicked 'Next page'. Waiting for page {page_num + 1} to load...")
            time.sleep(random.uniform(2.0, 3.5))

            page_num += 1

        except (StaleElementReferenceException, TimeoutException, Exception) as e_click:
            break 

    return player_transfer_values

# --- Processor Functions ---
def get_valid_players_from_part1(config: Dict[str, Any]) -> Set[str]:
    valid_players: Set[str] = set()
    part1_file_path = config['paths']['part1_results_absolute']
    minutes_col = config['processing']['part1_minutes_column']
    player_col = config['processing']['part1_player_column']
    min_minutes = config['processing']['min_minutes_threshold']

    df_part1 = pd.read_csv(part1_file_path)

    required_cols = [minutes_col, player_col]
    if not all(col in df_part1.columns for col in required_cols):
        return valid_players

    df_part1['minutes_numeric'] = pd.to_numeric(df_part1[minutes_col].astype(str).str.replace(',', ''), errors='coerce')
    df_part1.dropna(subset=['minutes_numeric'], inplace=True)
    df_filtered = df_part1[df_part1['minutes_numeric'].astype(int) > min_minutes]

    if not df_filtered.empty:
        valid_players = set(df_filtered[player_col].astype(str).str.strip().unique())
        print(f"Found {len(valid_players)} players with > {min_minutes} minutes.")
    else:
        print(f"No players found with > {min_minutes} minutes.")
    
    return valid_players

def process_transfer_data(scraped_data: PlayerData, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    if not scraped_data:  
        return None

    df_raw = pd.DataFrame(list(scraped_data.values()))

    if df_raw.empty: 
        return df_raw 

    save_data(df_raw, config['paths']['raw_data_full_path'], config['paths']['output_folder'])

    valid_player_set = get_valid_players_from_part1(config)

    if not valid_player_set:
        return df_raw
    
    df_raw['Player_Normalized'] = df_raw['Player'].astype(str).str.strip()
    df_filtered = df_raw[df_raw['Player_Normalized'].isin(valid_player_set)].copy()
    df_filtered.drop(columns=['Player_Normalized'], inplace=True)

    return df_filtered 

# --- Main Execution ---
def run_main_workflow(config: Dict[str, Any]):
    start_time = time.time()
    final_df = None
    with managed_driver(config) as driver:
        if not driver:
            return 

        scraped_data = scrape_transfer_data(driver, config)

        if scraped_data is not None: 
            final_df = process_transfer_data(scraped_data, config)
        
    # --- Save Final Processed Data ---
    if final_df is not None: 
        if not final_df.empty:
            transfer_saved = save_data(final_df, config['paths']['transfer_value_output_full_path'], config['paths']['output_folder'])
            estimation_saved = save_data(final_df, config['paths']['estimation_ready_data_full_path'], config['paths']['output_folder'])
            if transfer_saved and estimation_saved:
                 print(f"\nSuccessfully processed and saved filtered data. Final shape: {final_df.shape}")
            else:
                 print("\nProcessing complete, but failed to save one or both output files.")

    print(f"\n--- Workflow finished in {time.time() - start_time:.2f} seconds ---")

# --- Script Entry Point ---
if __name__ == '__main__':
    CONFIG = load_config() 

    CONFIG['paths']['part1_results_absolute'] = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['paths']['part1_results_relative']))
    CONFIG['paths']['raw_data_full_path'] = os.path.join(CONFIG['paths']['output_folder'], CONFIG['paths']['raw_data_filename'])
    CONFIG['paths']['transfer_value_output_full_path'] = os.path.join(CONFIG['paths']['output_folder'], CONFIG['paths']['transfer_value_output_filename'])
    CONFIG['paths']['estimation_ready_data_full_path'] = os.path.join(CONFIG['paths']['output_folder'], CONFIG['paths']['estimation_ready_data_filename'])

    # --- Pre-check for Input File ---
    part1_path = CONFIG['paths']['part1_results_absolute']
    if not os.path.exists(part1_path):
            print("="*50 + f"\nWARNING: Input file not found:\n'{part1_path}'\nProcessing step might fail or yield no filtered data.\n" + "="*50)
            time.sleep(2)

    # --- Run Workflow ---
    run_main_workflow(CONFIG)