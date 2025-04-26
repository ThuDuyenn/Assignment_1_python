import pandas as pd
import numpy as np # Cần numpy để kiểm tra kiểu số
import os

# --- Config ---
TEAM_COL = 'Team' # Tên cột chứa thông tin đội
# Không cần STATS_TO_ANALYZE nữa, sẽ tự động phát hiện
CSV_RELATIVE_PATH = os.path.join('..', 'problem_1', 'results.csv')

# --- Main Analysis Function ---
def analyze_top_teams_all_numeric(csv_path: str, team_column: str):
    """
    Phân tích dữ liệu từ CSV để tìm đội có chỉ số cao nhất cho TẤT CẢ các cột số.

    Args:
        csv_path: Đường dẫn đến file CSV.
        team_column: Tên cột chứa tên đội.

    Returns:
        Một dictionary chứa kết quả phân tích, với key là tên chỉ số (cột số)
        và value là tuple (tên đội dẫn đầu, giá trị cao nhất).
        Trả về None nếu có lỗi xảy ra.
    """
    try:
        # Xác định đường dẫn tuyệt đối dựa trên vị trí script (nếu có thể) hoặc CWD
        try:
            # Cố gắng lấy thư mục của script hiện tại
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_csv_path = os.path.join(current_script_dir, csv_path)
        except NameError:
            # __file__ không được định nghĩa khi chạy tương tác (vd: Jupyter)
            print("Cảnh báo: Không thể xác định thư mục script. Sử dụng đường dẫn tương đối với thư mục làm việc hiện tại.")
            absolute_csv_path = csv_path # Sử dụng đường dẫn tương đối ban đầu

        print(f"Đang thử tải dữ liệu từ: {absolute_csv_path}")
        # Đọc CSV, cố gắng suy luận kiểu dữ liệu tốt hơn
        df = pd.read_csv(absolute_csv_path)
        print(f"Tải dữ liệu thành công. Kích thước: {df.shape}")

        results = {}

        # Kiểm tra sự tồn tại của cột đội
        if team_column not in df.columns:
            print(f"Lỗi: Không tìm thấy cột đội '{team_column}' trong CSV.")
            return None

        # --- Tự động xác định các cột số cần phân tích ---
        potential_stats_cols = df.columns.drop(team_column) # Loại bỏ cột đội
        stats_to_analyze = []
        print("\nĐang xác định các cột số để phân tích...")
        for col in potential_stats_cols:
            # Thử chuyển đổi cột sang dạng số, ép lỗi thành NaN
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            # Chỉ phân tích nếu cột chứa ít nhất một giá trị số hợp lệ
            if not numeric_col.isnull().all():
                 # Kiểm tra xem kiểu dữ liệu gốc có phải là số không (loại trừ boolean)
                 # Hoặc nếu sau khi ép kiểu, nó trở thành số
                 # Điều này giúp bao gồm các cột số được lưu dưới dạng object/string
                 if pd.api.types.is_numeric_dtype(df[col].dtype) and not pd.api.types.is_bool_dtype(df[col].dtype):
                     stats_to_analyze.append(col)
                     print(f"  - Tìm thấy cột số (theo dtype): '{col}'")
                 elif not numeric_col.isnull().all(): # Nếu có giá trị số sau khi ép kiểu
                     # Kiểm tra thêm để chắc chắn nó không phải là cột chỉ chứa NaN sau khi ép kiểu
                     # (mặc dù kiểm tra isnull().all() ở trên đã làm điều này)
                     stats_to_analyze.append(col)
                     print(f"  - Tìm thấy cột có thể chuyển đổi sang số: '{col}'")
            else:
                 print(f"  - Bỏ qua cột '{col}' (không có dữ liệu số hợp lệ).")

        if not stats_to_analyze:
             print("Lỗi: Không tìm thấy cột số nào hợp lệ để phân tích trong tệp.")
             return None

        print(f"\nSẽ phân tích các cột: {', '.join(stats_to_analyze)}")
        # ----------------------------------------------------

        # --- Thực hiện phân tích cho các cột đã xác định ---
        for stat in stats_to_analyze:
            # Chuyển đổi cột thành số (lần nữa để đảm bảo an toàn)
            numeric_stat_col = pd.to_numeric(df[stat], errors='coerce')

            # Bỏ qua nếu không có dữ liệu số (kiểm tra lại)
            if numeric_stat_col.isnull().all():
                 results[stat] = ('No Numeric Data', None)
                 continue

            # Tìm chỉ số của hàng có giá trị lớn nhất
            # Quan trọng: Cần loại bỏ NaN trước khi tìm idxmax
            valid_numeric_col = numeric_stat_col.dropna()
            if valid_numeric_col.empty:
                 results[stat] = ('Only NaN Data', None)
                 continue
            idx_max = valid_numeric_col.idxmax()

            # Lấy giá trị lớn nhất và tên đội tương ứng
            max_value = df.loc[idx_max, stat] # Lấy giá trị gốc từ df để giữ nguyên định dạng
            # Chuyển đổi max_value sang số để so sánh chính xác
            numeric_max_value = pd.to_numeric(max_value, errors='coerce')

            # Lấy tất cả các đội có cùng giá trị max (so sánh trên dữ liệu số đã ép kiểu)
            # Cần xử lý NaN khi so sánh
            top_teams_df = df.loc[numeric_stat_col.notna() & (numeric_stat_col == numeric_max_value), team_column]
            top_teams_list = top_teams_df.unique().tolist()

            if len(top_teams_list) > 1:
                 results[stat] = (", ".join(top_teams_list), max_value) # Nối tên các đội
            elif len(top_teams_list) == 1:
                 results[stat] = (top_teams_list[0], max_value)
            else:
                 # Trường hợp không tìm thấy đội nào (rất hiếm, có thể do lỗi logic)
                 results[stat] = ('Error finding team', max_value)


            print(f"Đã phân tích '{stat}': Đội dẫn đầu = {results[stat][0]}, Giá trị = {results[stat][1]}")

        return results

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp tại '{absolute_csv_path}'.")
        print("Vui lòng đảm bảo tệp CSV tồn tại ở vị trí đó.")
        print(f"Thư mục làm việc hiện tại là: {os.getcwd()}")
        return None
    except pd.errors.EmptyDataError:
         print(f"Lỗi: Tệp '{absolute_csv_path}' trống.")
         return None
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn trong quá trình phân tích: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Main Execution ---
if __name__ == '__main__':
    print("Bắt đầu phân tích tất cả các cột số...")
    # Gọi hàm mới không cần truyền danh sách stats
    analysis_results = analyze_top_teams_all_numeric(CSV_RELATIVE_PATH, TEAM_COL)

    if analysis_results:
        print("\n--- Tóm tắt Phân tích (Tất cả các cột số) ---")
        for stat, (team, value) in analysis_results.items():
            # Kiểm tra các trường hợp đặc biệt trước
            if team == 'No Numeric Data':
                 print(f"* {stat}: Không có dữ liệu số hợp lệ trong cột.")
            elif team == 'Only NaN Data':
                 print(f"* {stat}: Chỉ chứa dữ liệu NaN sau khi chuyển đổi.")
            elif team == 'Error finding team':
                 print(f"* {stat}: Lỗi khi tìm đội dẫn đầu (Giá trị: {value}).")
            elif value is not None: # Các trường hợp thành công
                # Cố gắng định dạng giá trị số
                try:
                    numeric_value = pd.to_numeric(value)
                    if isinstance(numeric_value, (int, float, np.number)):
                         # Làm tròn số thực, giữ nguyên số nguyên
                         value_str = f"{numeric_value:.2f}" if isinstance(numeric_value, (float, np.floating)) else str(numeric_value)
                    else:
                         value_str = str(value) # Giữ nguyên nếu không phải số (dù không nên xảy ra)
                except:
                    value_str = str(value) # Giữ nguyên nếu không thể chuyển đổi

                print(f"* {stat}:")
                print(f"  - Đội dẫn đầu: {team}")
                print(f"  - Giá trị cao nhất: {value_str}")
            else:
                 print(f"* {stat}: Không thể phân tích (Giá trị là None).") # Trường hợp hiếm
        print("---------------------------------------------")

        # --- Đánh giá đội tốt nhất (dựa trên TẤT CẢ các chỉ số số) ---
        print("\n--- Đánh giá sơ bộ (Dựa trên TẤT CẢ các chỉ số số đã phân tích) ---")
        leader_counts = {}
        valid_stats_count = 0
        for stat, (team_str, value) in analysis_results.items():
             # Chỉ đếm các kết quả hợp lệ
             if value is not None and team_str not in ['No Numeric Data', 'Only NaN Data', 'Error finding team']:
                 valid_stats_count += 1
                 teams = team_str.split(", ") # Tách các đội nếu có nhiều đội dẫn đầu
                 for t in teams:
                     leader_counts[t] = leader_counts.get(t, 0) + 1

        if leader_counts:
            # Tìm đội dẫn đầu nhiều chỉ số nhất
            most_leads_team = max(leader_counts, key=leader_counts.get)
            num_leads = leader_counts[most_leads_team]
            print(f"Đã phân tích {valid_stats_count} chỉ số số hợp lệ.")
            print(f"Đội dẫn đầu nhiều chỉ số nhất: {most_leads_team} ({num_leads} chỉ số)")

            print("\nLưu ý quan trọng:")
            print(" - Đánh giá này dựa trên việc dẫn đầu SỐ LƯỢNG chỉ số, không phải CHẤT LƯỢNG hay TẦM QUAN TRỌNG của chỉ số đó.")
            print(" - Một số cột số (ví dụ: ID, năm, thứ hạng) có thể không phản ánh hiệu suất thực tế của đội.")
            print(" - Kết quả này KHÔNG THAY THẾ cho bảng xếp hạng chính thức dựa trên điểm số.")
        else:
            print("Không có đủ dữ liệu hợp lệ để đưa ra đánh giá.")
        print("--------------------------------------------------------------------")

    else:
        print("\nKhông thể hoàn thành phân tích do có lỗi.")
