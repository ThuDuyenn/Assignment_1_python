import pandas as pd
import numpy as np
import os
import json 

def load_config(config_path='config.json'):
    """Đọc file cấu hình JSON và trả về dictionary chứa cấu hình."""
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

# --- Hàm Phân tích Chính (Giờ nhận config_data dict và các Set) ---
def analyze_performance_by_stat_type(df, team_column, ignore_set, bad_set, good_set):
    results = {}
    classified_stats = {'good': [], 'bad': [], 'ignore': [], 'uncategorized': []}
    stats_to_analyze = []

    potential_stats_cols = df.columns.drop(team_column, errors='ignore')

    for col in potential_stats_cols:
        stat_type = classify_stat(col, ignore_set, bad_set, good_set)
        classified_stats[stat_type].append(col)
        if stat_type != 'ignore':
            temp_numeric_col = pd.to_numeric(df[col], errors='coerce')
            if not temp_numeric_col.isnull().all():
                 stats_to_analyze.append(col)

    print(f"\nTổng cộng: {len(classified_stats['good'])} GOOD, {len(classified_stats['bad'])} BAD, {len(classified_stats['ignore'])} IGNORE, {len(classified_stats['uncategorized'])} UNCATEGORIZED")
    print(f"Sẽ phân tích {len(stats_to_analyze)} cột (GOOD, BAD, UNCATEGORIZED có dữ liệu số).")
    print("---------------------------------------------")

    # --- Thực hiện phân tích ---
    for stat in stats_to_analyze:
        stat_type = classify_stat(stat, ignore_set, bad_set, good_set)
        numeric_stat_col = pd.to_numeric(df[stat], errors='coerce')

        if numeric_stat_col.isnull().all():
            results[stat] = ('Only NaN Data', None, stat_type)
            continue

        valid_numeric_col = numeric_stat_col.dropna()
    
        best_value = None
        best_value_idx = None

        if stat_type == 'good' or stat_type == 'uncategorized':
            best_value_idx = valid_numeric_col.idxmax()
            best_value = valid_numeric_col.max()
        elif stat_type == 'bad':
            best_value_idx = valid_numeric_col.idxmin()
            best_value = valid_numeric_col.min()

        original_value_display = df.loc[best_value_idx, stat]
        top_teams_df = df.loc[numeric_stat_col.notna() & (numeric_stat_col == best_value), team_column]
        top_teams_list = top_teams_df.unique().tolist()
        top_str = ", ".join(top_teams_list) if top_teams_list else "Không tìm thấy"
        results[stat] = (top_str, original_value_display, stat_type)

    return results

def format_value(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = pd.to_numeric(value)
    if pd.api.types.is_integer_dtype(type(numeric_value)) or (pd.api.types.is_float_dtype(type(numeric_value)) and numeric_value == np.floor(numeric_value)):
            return str(int(numeric_value))
    elif pd.api.types.is_float_dtype(type(numeric_value)):
            return f"{numeric_value:.2f}"
    else:
            return str(value)

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- 1. Đọc cấu hình từ JSON ---
    CONFIG_FILE_PATH = 'config.json' 
    config_data = load_config(CONFIG_FILE_PATH)

    team_col_name = config_data.get('team_column')
    csv_relative_path = config_data.get('csv_relative_path')
    na_vals = config_data.get('na_values', []) 

    ignore_set = config_data.get('ignore_stats_set', set())
    bad_set = config_data.get('bad_stats_set', set())
    good_set = config_data.get('good_stats_set', set())

    # --- 2. Xác định đường dẫn tuyệt đối và đọc CSV ---

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_FULL_PATH = os.path.normpath(os.path.join(current_script_dir, csv_relative_path))
    dataframe = pd.read_csv(CSV_FULL_PATH, na_values=na_vals)
    
    # --- 3. Thực hiện phân tích ---
    analysis_results = analyze_performance_by_stat_type(
        df=dataframe,
        team_column=team_col_name,
        ignore_set=ignore_set,
        bad_set=bad_set,
        good_set=good_set
    )

    # --- 4. Hiển thị kết quả và đánh giá ---
    if analysis_results:
        print("\n--- Tóm tắt Phân tích Chi tiết ---")
        for stat, (team, value, stat_type) in analysis_results.items():
            value_str = format_value(value)
            stat_type_hr = stat_type.upper()
            print(f"* {stat} [{stat_type_hr}]:")
            if team == 'Only NaN Data':
                print(f"  - Trạng thái: {team}")
            elif team == "Không tìm thấy":
                 print(f"  - Không tìm thấy đội dẫn đầu.")
            else:
                analysis_desc = ""
                if stat_type == 'good': analysis_desc = "cao nhất (tốt)"
                elif stat_type == 'bad': analysis_desc = "thấp nhất (tốt)"
                elif stat_type == 'uncategorized': analysis_desc = "cao nhất (chưa phân loại)"
                value_display = f"  - Giá trị: {value_str}" if value_str != "N/A" else "  - Giá trị: (Không có)"
                print(f"  - Đội có giá trị {analysis_desc}: {team}")
                print(value_display)
        print("-----------------------------------")

        print("\n--- Đánh giá Sơ bộ (Dựa trên số lượng chỉ số dẫn đầu) ---")
        good_lead_counts = {}
        bad_lead_counts = {}
        valid_stats_count = 0
        for stat, (team_str, value, stat_type) in analysis_results.items():
             if value is not None and pd.notna(value) and team_str not in ['Only NaN Data', 'Không tìm thấy']:
                 valid_stats_count += 1
                 teams = team_str.split(", ")
                 for t in teams:
                     if stat_type == 'good':
                         good_lead_counts[t] = good_lead_counts.get(t, 0) + 1
                     elif stat_type == 'bad':
                         bad_lead_counts[t] = bad_lead_counts.get(t, 0) + 1

        if good_lead_counts or bad_lead_counts:
            print(f"Đã phân tích và tìm thấy người dẫn đầu cho {valid_stats_count} chỉ số số hợp lệ (Good/Bad).")
            print("\nSố lần dẫn đầu chỉ số 'TỐT' (Cao nhất trong nhóm 'Good'):")
            if good_lead_counts:
                sorted_good = sorted(good_lead_counts.items(), key=lambda item: item[1], reverse=True)
                for team, count in sorted_good: print(f"  - {team}: {count} lần")
            else: print("  (Không có)")
            print("\nSố lần 'dẫn đầu' chỉ số 'XẤU' (Thấp nhất trong nhóm 'Bad' - cũng là Tốt):")
            if bad_lead_counts:
                sorted_bad = sorted(bad_lead_counts.items(), key=lambda item: item[1], reverse=True)
                for team, count in sorted_bad: print(f"  - {team}: {count} lần")
            else: print("  (Không có)")
            combined_scores = {}
            all_teams = set(good_lead_counts.keys()) | set(bad_lead_counts.keys())
            for team in all_teams: combined_scores[team] = good_lead_counts.get(team, 0) + bad_lead_counts.get(team, 0)
            
            if combined_scores:
                sorted_combined = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
                if sorted_combined:
                    best_overall_team, best_score = sorted_combined[0]
                    print(f"\n-> Đội dẫn đầu nhiều nhất: {best_overall_team} ({best_score} lần)")
            else:
                print("\nKhông có đội nào dẫn đầu các chỉ số 'Good' hoặc 'Bad' đã phân tích.")
        else: print("Không có đủ dữ liệu hợp lệ để đưa ra đánh giá tổng hợp.")
    else:
        print("\nPhân tích không thành công hoặc không có kết quả.")