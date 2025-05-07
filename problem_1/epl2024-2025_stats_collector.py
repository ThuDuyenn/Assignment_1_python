import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import json
from collections import defaultdict

CONFIG_FILE = 'config.json'

def load_config(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

config = load_config(CONFIG_FILE)

BASE_URL = config["base_url"]
STATS_CONFIG = config["stats_config"]
COLUMN_ORDER = config["column_order"]
CSV_FILENAME = config["csv_filename"]
CSV_FILENAME_1 = "results1.csv"
MIN_MINUTES = config["min_minutes"] 
WAIT_TIMEOUT = config["wait_timeout"]
CURRENT_SEASON = "2024-2025"

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

def get_nation_code(nation_cell, default="N/a"):
    if not nation_cell:
        return default
    outer_span = nation_cell.select_one('a > span')
    if outer_span:
        direct_texts = [text.strip() for text in outer_span.find_all(text=True, recursive=False) if text.strip()]
        if direct_texts:
            code = direct_texts[-1]
            if len(code) == 3 and code.isupper():
                return code
    full_text = nation_cell.get_text(strip=True)
    parts = full_text.split()
    nation_code = next((part for part in parts if part.isupper()), None)
    return nation_code if nation_code else (full_text if full_text else default)

def format_value(stat_name, value):
    if value is None or value == '':
        return "N/a"
    str_value = str(value)
    if str_value != "N/a":
        if stat_name == 'Age':
            return str_value.split('-')[0] if '-' in str_value else str_value
        elif ',' in str_value:
            if re.match(r'^-?[\d,]+(\.\d+)?$', str_value):
                return str_value.replace(',', '')
    return str_value

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

        if not player_cell:
             continue

        player_name = player_cell.get_text(strip=True)
        player_id = get_player_id(player_cell)
        team_name = team_cell.get_text(strip=True) if team_cell else "Unknown"

        primary_id = player_id if player_id else player_name
        if not primary_id:
             continue

        player_team_key = (primary_id, team_name)

        if player_team_key not in player_data:
            player_data[player_team_key] = {}
            for col in COLUMN_ORDER:
                player_data[player_team_key][col] = None
            player_data[player_team_key]['player_id'] = player_id
            player_data[player_team_key]['Name'] = player_name
            player_data[player_team_key]['Team'] = team_name

        nation_cell = row.find('td', attrs={'data-stat': 'nationality'})
        if nation_cell and player_data[player_team_key].get('Nation') is None:
             nation_code = get_nation_code(nation_cell, default=None)
             if nation_code:
                 player_data[player_team_key]['Nation'] = nation_code

        for data_stat_attr in stats_to_extract:
            if data_stat_attr == 'nationality':
                continue

            stat_key = None
            for key, cfg in global_stats_config.items():
                if cfg['data_stat'] == data_stat_attr and cfg['table_id'] == table_id:
                    if key != 'Nation':
                        stat_key = key
                        break

            if stat_key:
                value = safe_get_text(row, data_stat_attr, default=None)
                if value is not None or player_data[player_team_key].get(stat_key) is None:
                     player_data[player_team_key][stat_key] = value

if __name__ == "__main__":
    scraped_data = {}
    configs_by_url = {}

    for csv_col, cfg in STATS_CONFIG.items():
        full_url = f"{BASE_URL}{CURRENT_SEASON}/{cfg['url_suffix']}"
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
            scrape_fbref_data(driver, url, table_id, list(stats_set), scraped_data, STATS_CONFIG)
        time.sleep(1)

    driver.quit()

    # --- Processing for results1.csv (single team > 90 minutes) ---
    single_team_over_90_list = []
    for player_team_key, stats_dict in scraped_data.items():
        minutes_raw = stats_dict.get('Playing Time: minutes', '0')
        minutes_str = str(minutes_raw).replace(',', '') if minutes_raw is not None else '0'
        minutes_played_this_team = int(minutes_str) if minutes_str.isdigit() else 0

        if minutes_played_this_team > MIN_MINUTES:
            formatted_player_dict = {}
            for col_name in COLUMN_ORDER:
                raw_or_processed_value = stats_dict.get(col_name, None)
                formatted_val = format_value(col_name, raw_or_processed_value)
                formatted_player_dict[col_name] = formatted_val
            single_team_over_90_list.append(formatted_player_dict)

    # --- Processing for results.csv (total minutes > MIN_MINUTES) ---
    player_minutes_aggregate = defaultdict(lambda: {'total_minutes': 0, 'entries': []})

    for player_team_key, stats_dict in scraped_data.items():
        player_unique_id = stats_dict.get('player_id') or stats_dict.get('Name')
        if not player_unique_id: continue

        minutes_raw = stats_dict.get('Playing Time: minutes', '0')
        minutes_str = str(minutes_raw).replace(',', '') if minutes_raw is not None else '0'
        minutes_played = int(minutes_str) if minutes_str.isdigit() else 0

        player_minutes_aggregate[player_unique_id]['total_minutes'] += minutes_played
        player_minutes_aggregate[player_unique_id]['entries'].append(stats_dict)

    final_player_list_of_dicts = []
    for player_id, agg_data in player_minutes_aggregate.items():
        if agg_data['total_minutes'] > MIN_MINUTES:
            for stats_dict in agg_data['entries']:
                formatted_player_dict = {}
                for col_name in COLUMN_ORDER:
                    raw_or_processed_value = stats_dict.get(col_name, None)
                    formatted_val = format_value(col_name, raw_or_processed_value)
                    formatted_player_dict[col_name] = formatted_val
                final_player_list_of_dicts.append(formatted_player_dict)

    # --- Sorting and Saving results.csv ---
    if final_player_list_of_dicts:
         final_player_list_of_dicts.sort(key=lambda p_dict: str(p_dict.get('Name', '')).split()[0] if p_dict.get('Name') and ' ' in str(p_dict.get('Name', '')) else str(p_dict.get('Name', '')))
         print("Sorting by first name applied for results.csv.")
         df = pd.DataFrame(final_player_list_of_dicts)
         df = df[COLUMN_ORDER]
         df.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
         print(f"Data saved successfully to {CSV_FILENAME}")
    else:
         print(f"No players met the criteria for {CSV_FILENAME} or data could not be processed.")

    # --- Sorting and Saving results1.csv ---
    if single_team_over_90_list:
        single_team_over_90_list.sort(key=lambda p_dict: str(p_dict.get('Name', '')).split()[0] if p_dict.get('Name') and ' ' in str(p_dict.get('Name', '')) else str(p_dict.get('Name', '')))
        print("Sorting by first name applied for results1.csv.")
        df1 = pd.DataFrame(single_team_over_90_list)
        df1 = df1[COLUMN_ORDER]
        df1.to_csv(CSV_FILENAME_1, index=False, encoding='utf-8-sig')
        print(f"Data saved successfully to {CSV_FILENAME_1}")
    else:
        print(f"No players met the criteria (> {MIN_MINUTES} minutes for a single team) for {CSV_FILENAME_1}.")