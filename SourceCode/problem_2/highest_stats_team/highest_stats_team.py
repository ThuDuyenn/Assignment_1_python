import pandas as pd
import numpy as np
import os
import json

def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    config_data['ignore_stats_set'] = {k.lower() for k in config_data.get('ignore_stats_keywords', [])}
    config_data['bad_stats_set'] = {k.lower() for k in config_data.get('bad_stats_keywords', [])}
    config_data['good_stats_set'] = {k.lower() for k in config_data.get('good_stats_keywords', [])}
    return config_data

def classify_stat(stat_name, ignore_set, bad_set, good_set):
    stat_lower = stat_name.lower()
    if stat_lower in ignore_set:
        return 'ignore'
    if stat_lower in bad_set:
        return 'bad'
    if stat_lower in good_set:
        return 'good'
    return 'uncategorized'

# --- Hàm Phân tích Chính ---
def analyze_performance_by_stat_type(df, team_column, ignore_set, bad_set, good_set):
    results = {}
    stats_to_analyze = []
    potential_stats_cols = [col for col in df.columns if col.startswith("Mean of ") and col != team_column]

    for col_full_name in potential_stats_cols:
        actual_stat_name = col_full_name.replace("Mean of ", "", 1)
        stat_type = classify_stat(actual_stat_name, ignore_set, bad_set, good_set)
        if stat_type != 'ignore':
            temp_numeric_col = pd.to_numeric(df[col_full_name], errors='coerce')
            if not temp_numeric_col.isnull().all():
                    stats_to_analyze.append(col_full_name)

    print(f"Sẽ phân tích {len(stats_to_analyze)} cột 'Mean of...' (GOOD, BAD, UNCATEGORIZED có dữ liệu số).")
    print("---------------------------------------------")

    # --- Thực hiện phân tích ---
    for stat_col_full_name in stats_to_analyze: 
        actual_stat_name_key = stat_col_full_name.replace("Mean of ", "", 1) 
        stat_type = classify_stat(actual_stat_name_key, ignore_set, bad_set, good_set)
        
        numeric_stat_col = pd.to_numeric(df[stat_col_full_name], errors='coerce')
        valid_numeric_col = numeric_stat_col.dropna()
            
        best_value = None
        best_value_idx = None
        
        if stat_type == 'good' or stat_type == 'uncategorized':
            best_value_idx = valid_numeric_col.idxmax()
            best_value = valid_numeric_col.max()
        elif stat_type == 'bad':
            best_value_idx = valid_numeric_col.idxmin()
            best_value = valid_numeric_col.min()
        
        if best_value_idx is None or best_value is None: 
            results[actual_stat_name_key] = ('Error finding best value', None, stat_type)
            continue

        original_value_display = df.loc[best_value_idx, stat_col_full_name]
        top_teams_df = df.loc[numeric_stat_col.notna() & (df[stat_col_full_name] == best_value), team_column]
        top_teams_list = top_teams_df.unique().tolist()
        
        cleaned_top_teams_list = [str(t).strip() for t in top_teams_list if str(t).strip()]
        top_str = ", ".join(cleaned_top_teams_list) if cleaned_top_teams_list else "Không tìm thấy"
        
        results[actual_stat_name_key] = (top_str, original_value_display, stat_type)

    return results

def format_value(value):
    if value is None or pd.isna(value):
        return "N/A"
    try:
        numeric_value = pd.to_numeric(value)
        if np.isinf(numeric_value):
            return str(numeric_value)
        if isinstance(numeric_value, (int, np.integer)) or \
           (isinstance(numeric_value, (float, np.floating)) and numeric_value == np.floor(numeric_value)):
            return str(int(numeric_value))
        elif isinstance(numeric_value, (float, np.floating)):
            return f"{numeric_value:.2f}"
        else:
            return str(value)
    except ValueError:
        return str(value)

# --- Main Execution Block ---
if __name__ == '__main__':
    CONFIG_FILE_PATH = 'config.json'
    config_data = load_config(CONFIG_FILE_PATH)

    na_vals = config_data.get('na_values', [])
    ignore_set = config_data.get('ignore_stats_set', set())
    bad_set = config_data.get('bad_stats_set', set())
    good_set = config_data.get('good_stats_set', set())

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path_relative_to_script_dir = os.path.join('..', 'calculating_statistics', 'results2.csv')
    CSV_FULL_PATH = os.path.normpath(os.path.join(current_dir, csv_path_relative_to_script_dir))
    
    dataframe = pd.read_csv(CSV_FULL_PATH, na_values=na_vals)

    team_col_name = dataframe.columns[0]

    if team_col_name in dataframe.columns:
        dataframe[team_col_name] = dataframe[team_col_name].astype(str).str.lower()
        initial_row_count = len(dataframe)
        dataframe = dataframe[dataframe[team_col_name] != 'all']
        
    analysis_results = analyze_performance_by_stat_type(
        df=dataframe,
        team_column=team_col_name,
        ignore_set=ignore_set,
        bad_set=bad_set,
        good_set=good_set
    )

    if analysis_results:
        print("\n--- Tóm tắt Phân tích Chi tiết ---")
        sorted_analysis_results = sorted(analysis_results.items())

        for stat_actual_name, (team, value, stat_type) in sorted_analysis_results:
            value_str = format_value(value)
            stat_type_hr = stat_type.upper()
            print(f"* {stat_actual_name} [{stat_type_hr}]:")
            if team == "Không tìm thấy" or team == "Error finding best value":
                print(f"  - Trạng thái: {team}")
            else:
                analysis_desc = ""
                if stat_type == 'good': analysis_desc = "cao nhất (tốt)"
                elif stat_type == 'bad': analysis_desc = "thấp nhất (tốt)"
                elif stat_type == 'uncategorized': analysis_desc = "cao nhất (chưa phân loại)"
                
                print(f"  - Đội có giá trị {analysis_desc}: {team}")
                value_display = f"  - Giá trị (Mean): {value_str}" if value_str != "N/A" else "  - Giá trị (Mean): (Không có)"
                print(value_display)
        print("-----------------------------------")

        good_lead_counts = {}
        bad_lead_counts = {}
        valid_stats_count = 0
        for stat_actual_name, (team_str, value, stat_type) in analysis_results.items():
            if value is not None and pd.notna(value) and team_str not in ['Không tìm thấy', 'Error finding best value'] and team_str.strip():
                valid_stats_count += 1
                teams = team_str.split(", ")
                for t_raw in teams:
                    t = t_raw.strip()
                    if t: 
                        if stat_type == 'good':
                            good_lead_counts[t] = good_lead_counts.get(t, 0) + 1
                        elif stat_type == 'bad':
                            bad_lead_counts[t] = bad_lead_counts.get(t, 0) + 1
        
        # --- Bảng Xếp Hạng Tất Cả Các Đội ---
        # Khởi tạo điểm tổng hợp cho tất cả các đội còn lại trong DataFrame (sau khi bỏ hàng 'all')
        all_teams_in_filtered_df_raw = dataframe[team_col_name].dropna().unique()
        all_teams_cleaned = {str(team).strip() for team in all_teams_in_filtered_df_raw if str(team).strip()}

        combined_scores = {team: 0 for team in all_teams_cleaned}

        # Cập nhật điểm từ good_lead_counts và bad_lead_counts
        for team, count in good_lead_counts.items():
            if team in combined_scores: combined_scores[team] += count
        for team, count in bad_lead_counts.items():
            if team in combined_scores: combined_scores[team] += count
        
        if combined_scores:
            # Sắp xếp các đội theo điểm tổng hợp (giảm dần), sau đó theo tên đội (tăng dần)
            sorted_combined_ranking = sorted(combined_scores.items(), key=lambda x: (-x[1], x[0]))
            
            print("\n--- Bảng Xếp Hạng Tổng Hợp Các Đội (Dựa trên số lần dẫn đầu 'Mean of...') ---")
            if not sorted_combined_ranking:
                print("Không có dữ liệu để xếp hạng.")
            else:
                current_rank_display = 0
                last_score_val = -1 
                for i, (team_name, team_score) in enumerate(sorted_combined_ranking):
                    if team_score != last_score_val: 
                        current_rank_display = i + 1
                        last_score_val = team_score
                    print(f"Hạng {current_rank_display}: {team_name} ({team_score} lần dẫn đầu)")
        else:
            print("\nKhông có đội nào dẫn đầu các chỉ số 'Mean of...' (Good/Bad) đã phân tích để xếp hạng.")
        # --- Kết thúc Bảng Xếp Hạng ---