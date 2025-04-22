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
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

config = load_config(CONFIG_FILE)

BASE_URL = config["base_url"]
SEASON_URL_SUFFIX = config["season_url_suffix"]
STATS_CONFIG = config["stats_config"] 
COLUMN_ORDER = config["column_order"] 
CSV_FILENAME = config["csv_filename"]
MIN_MINUTES = config["min_minutes"]
WAIT_TIMEOUT = config["wait_timeout"]

def safe_get_text(element, data_stat, default="N/a"):
    cell = element.find('td', attrs={'data-stat': data_stat})
    text = cell.get_text(strip=True) if cell else ''
    return text if text else default

def get_player_id(player_cell):
    link = player_cell.find('a')
    if link and link.has_attr('href'):
        match = re.search(r'/players/([a-zA-Z0-9]+)/', link['href'])
        if match:
            return match.group(1)
    return None

def format_value(stat_name, value):
    if value is None or value == '':
        return "N/a"
    if value != "N/a":
        if stat_name == 'Nation':
            return value.split()[0] if value and ' ' in value else value 
        elif stat_name == 'Age':
            return value.split('-')[0] if '-' in value else value
        elif isinstance(value, str) and ',' in value:
            if re.match(r'^-?[\d,]+(\.\d+)?$', value):
                return value.replace(',', '')
    return value

# --- Main Scraping Function ---
def scrape_fbref_data(driver, url, table_id, stats_to_extract, player_data, global_stats_config):
    driver.get(url)
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"table#{table_id} tbody tr")))
    time.sleep(1)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    table = soup.find('table', id=table_id)
    rows = table.select('tbody tr:not(.thead)')

    for row in rows:
        player_cell = row.find('td', attrs={'data-stat': 'player'})
        team_cell = row.find('td', attrs={'data-stat': 'team'}) 
        if not player_cell or not team_cell:
            continue 

        player_id = get_player_id(player_cell)
        player_name = player_cell.get_text(strip=True)
        team_name = team_cell.get_text(strip=True) 

        primary_id = player_id if player_id else player_name
        if not primary_id:
             continue

        player_team_key = (primary_id, team_name) 

        if player_team_key not in player_data:
            player_data[player_team_key] = {
                'player_id': player_id,
                'Name': player_name,
                'Team': team_name
            }
        for stat_name in stats_to_extract:
            stat_key = None
            for key, cfg in global_stats_config.items():
                if cfg['data_stat'] == stat_name and cfg['table_id'] == table_id:
                    stat_key = key
                    break
            if stat_key:
                value = safe_get_text(row, stat_name, default="N/a")
                if stat_key != 'Team' or player_data[player_team_key].get('Team') is None:
                     player_data[player_team_key][stat_key] = value
# --- Main Execution ---

if __name__ == "__main__":
    start_time = time.time()
    all_players_data = {}

    configs_by_url = {}
    for csv_col, cfg in STATS_CONFIG.items(): 
        full_url = BASE_URL + cfg['url_suffix'] + SEASON_URL_SUFFIX
        table_id = cfg['table_id']
        data_stat = cfg['data_stat']

        if full_url not in configs_by_url:
            configs_by_url[full_url] = {}
        if table_id not in configs_by_url[full_url]:
            configs_by_url[full_url][table_id] = set()

        configs_by_url[full_url][table_id].add(data_stat)

    driver = webdriver.Chrome()

    for url, tables in configs_by_url.items():
        for table_id, stats_set in tables.items():
            scrape_fbref_data(driver, url, table_id, list(stats_set), all_players_data, STATS_CONFIG)
            time.sleep(1)

    driver.quit()

    # --- Data Processing & Filtering ---
    final_data_list = []
    if all_players_data:
        players_to_process = list(all_players_data.values())

        for stats_dict in players_to_process:
            minutes_raw = stats_dict.get('Playing Time: minutes', '0')
            minutes_str = str(minutes_raw).replace(',', '') if minutes_raw is not None else '0'
            minutes_played = int(minutes_str) if minutes_str.isdigit() else 0

            if minutes_played > MIN_MINUTES:
                player_row = []
                for col_name in COLUMN_ORDER: 
                    raw_value = stats_dict.get(col_name, None)
                    formatted_val = format_value(col_name, raw_value)
                    player_row.append(formatted_val)
                final_data_list.append(player_row)

        if final_data_list:
            final_data_list.sort(key=lambda x: x[0])

    # --- Save to CSV ---
    df = pd.DataFrame(final_data_list, columns=COLUMN_ORDER)
    df.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
    print("Data saved successfully.")