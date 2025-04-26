import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình ---
# *** THAY ĐỔI: Sử dụng cột 'Name' thay vì 'Player' ***
PLAYER_COL = 'Name'
# Đường dẫn tương đối đến file CSV (giống các script trước)
CSV_RELATIVE_PATH = os.path.join('..', 'problem_1', 'results.csv')
# Số cụm tối đa để thử nghiệm bằng phương pháp Elbow
MAX_K_TO_TEST = 10
# Thư mục lưu biểu đồ Elbow
OUTPUT_DIR = 'kmeans_analysis'

# --- Hàm trợ giúp ---
def ensure_dir(path: str):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)

# --- Hàm chính ---
def cluster_players_kmeans(csv_path: str, player_col: str, max_k: int = 10):
    """
    Thực hiện phân cụm cầu thủ bằng K-Means.

    Args:
        csv_path: Đường dẫn đến file CSV.
        player_col: Tên cột chứa định danh cầu thủ (đã cập nhật thành 'Name').
        max_k: Số cụm tối đa để kiểm tra bằng phương pháp Elbow.

    Returns:
        DataFrame chứa thông tin cầu thủ và nhãn cụm được gán,
        hoặc None nếu có lỗi.
    """
    try:
        # --- 1. Nạp dữ liệu ---
        try:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_csv_path = os.path.join(current_script_dir, csv_path)
        except NameError:
            print("Cảnh báo: Không thể xác định thư mục script. Sử dụng đường dẫn tương đối với thư mục làm việc hiện tại.")
            absolute_csv_path = csv_path

        print(f"Đang thử tải dữ liệu từ: {absolute_csv_path}")
        df = pd.read_csv(absolute_csv_path)
        print(f"Tải dữ liệu thành công. Kích thước: {df.shape}")

        if player_col not in df.columns:
            # Thông báo lỗi cập nhật theo tên cột mới
            print(f"Lỗi: Không tìm thấy cột cầu thủ '{player_col}' trong CSV.")
            return None

        # --- 2. Chuẩn bị dữ liệu ---
        print("\nĐang chuẩn bị dữ liệu...")
        # Chọn các cột số để phân cụm
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
             print("Lỗi: Không tìm thấy cột số nào để thực hiện phân cụm.")
             return None

        print(f"Các cột số được chọn để phân cụm: {', '.join(numeric_cols)}")
        features_df = df[numeric_cols].copy()

        # Xử lý giá trị thiếu (NaN) bằng cách thay thế bằng giá trị trung bình của cột
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features_df)
        features_imputed_df = pd.DataFrame(features_imputed, columns=numeric_cols, index=df.index)
        print(f"Đã xử lý giá trị thiếu bằng cách thay thế bằng giá trị trung bình.")

        # --- 3. Chuẩn hóa dữ liệu ---
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed_df)
        features_scaled_df = pd.DataFrame(features_scaled, columns=numeric_cols, index=df.index)
        print("Đã chuẩn hóa dữ liệu bằng StandardScaler.")

        # --- 4. Xác định số cụm tối ưu (K) bằng phương pháp Elbow ---
        print(f"\nĐang tìm số cụm tối ưu (K) bằng phương pháp Elbow (tối đa K={max_k})...")
        wcss = [] # Within-Cluster Sum of Squares
        possible_k_values = range(1, max_k + 1)

        for k in possible_k_values:
            # Sử dụng n_init='auto' thay vì 10 để tránh cảnh báo trong phiên bản sklearn mới
            kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
            kmeans_model.fit(features_scaled_df)
            wcss.append(kmeans_model.inertia_)

        # Vẽ biểu đồ Elbow
        ensure_dir(OUTPUT_DIR)
        elbow_plot_path = os.path.join(OUTPUT_DIR, 'kmeans_elbow_plot.png')
        plt.figure(figsize=(10, 6))
        plt.plot(possible_k_values, wcss, marker='o', linestyle='--')
        plt.title('Phương pháp Elbow để tìm số cụm tối ưu (K)')
        plt.xlabel('Số lượng cụm (K)')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.xticks(possible_k_values)
        plt.grid(True)
        try:
            plt.savefig(elbow_plot_path)
            print(f"Đã lưu biểu đồ Elbow vào: {elbow_plot_path}")
            print("=> Hãy xem biểu đồ Elbow và chọn giá trị K tại điểm 'khuỷu tay'.")
        except Exception as e:
            print(f"Lỗi khi lưu biểu đồ Elbow: {e}")
        finally:
            plt.close()

        # --- Tạm dừng để người dùng nhập K tối ưu ---
        optimal_k = None
        while optimal_k is None:
            try:
                k_input = input(f"Dựa vào biểu đồ Elbow, hãy nhập số cụm tối ưu (K) bạn chọn (từ 1 đến {max_k}): ")
                k_chosen = int(k_input)
                if 1 <= k_chosen <= max_k:
                    optimal_k = k_chosen
                else:
                    print(f"Vui lòng nhập một số nguyên từ 1 đến {max_k}.")
            except ValueError:
                print("Đầu vào không hợp lệ. Vui lòng nhập một số nguyên.")

        print(f"Bạn đã chọn K = {optimal_k}.")

        # --- 5. Áp dụng K-means với K tối ưu ---
        print(f"\nĐang áp dụng K-Means với K = {optimal_k}...")
        # Sử dụng n_init='auto'
        final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init='auto', random_state=42)
        final_kmeans.fit(features_scaled_df)

        # Lấy nhãn cụm cho từng cầu thủ
        cluster_labels = final_kmeans.labels_

        # --- 6. Thêm nhãn cụm vào DataFrame gốc ---
        df_clustered = df.copy()
        df_clustered['Cluster'] = cluster_labels
        print("Đã gán nhãn cụm cho các cầu thủ.")

        # --- 7. Phân tích kết quả (Tùy chọn nhưng hữu ích) ---
        print("\n--- Phân tích sơ bộ các cụm ---")
        # Tính giá trị trung bình của các chỉ số cho từng cụm
        # Đảm bảo chỉ lấy các cột số đã dùng để scale và fit
        cluster_centroids_original_scale = scaler.inverse_transform(final_kmeans.cluster_centers_)
        centroid_df = pd.DataFrame(cluster_centroids_original_scale, columns=features_imputed_df.columns) # Sử dụng cột từ features_imputed_df
        centroid_df.index.name = 'Cluster'
        print("Giá trị trung bình của các chỉ số cho mỗi cụm (đã quay về thang đo gốc):")
        # Sử dụng display để hiển thị DataFrame đẹp hơn nếu chạy trong môi trường hỗ trợ (Jupyter)
        try:
            from IPython.display import display
            display(centroid_df.round(2))
        except ImportError:
             print(centroid_df.round(2)) # In ra bình thường nếu không có IPython

        # Đếm số lượng cầu thủ trong mỗi cụm
        print("\nSố lượng cầu thủ trong mỗi cụm:")
        print(df_clustered['Cluster'].value_counts().sort_index())
        print("---------------------------------")

        # Lưu kết quả (tùy chọn)
        output_csv_path = os.path.join(OUTPUT_DIR, 'player_clusters.csv')
        try:
             df_clustered.to_csv(output_csv_path, index=False)
             print(f"Đã lưu kết quả phân cụm vào: {output_csv_path}")
        except Exception as e:
             print(f"Lỗi khi lưu file CSV kết quả: {e}")


        return df_clustered

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp tại '{absolute_csv_path}'.")
        return None
    except pd.errors.EmptyDataError:
         print(f"Lỗi: Tệp '{absolute_csv_path}' trống.")
         return None
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Main Execution ---
if __name__ == '__main__':
    print("Bắt đầu quá trình phân cụm cầu thủ bằng K-Means...")
    # Gọi hàm với PLAYER_COL đã được cập nhật thành 'Name'
    clustered_data = cluster_players_kmeans(CSV_RELATIVE_PATH, PLAYER_COL, MAX_K_TO_TEST)

    if clustered_data is not None:
        print("\nQuá trình phân cụm hoàn tất.")
        # Ví dụ: Hiển thị một vài cầu thủ từ mỗi cụm
        print("\nVí dụ một vài cầu thủ từ mỗi cụm:")
        # Lấy danh sách cột số thực tế đã dùng để phân cụm
        numeric_cols_used = clustered_data.select_dtypes(include=np.number).columns.drop('Cluster', errors='ignore').tolist()
        for cluster_id in sorted(clustered_data['Cluster'].unique()):
             print(f"\n--- Cụm {cluster_id} ---")
             # Lấy tối đa 5 cầu thủ từ cụm này
             sample_players = clustered_data[clustered_data['Cluster'] == cluster_id].head(5)
             # Hiển thị cột tên, tối đa 5 cột số đầu tiên đã dùng, và cột Cluster
             cols_to_show = [PLAYER_COL] + numeric_cols_used[:5] + ['Cluster']
             # Đảm bảo các cột này tồn tại trong sample_players trước khi in
             cols_to_show = [col for col in cols_to_show if col in sample_players.columns]
             print(sample_players[cols_to_show])
    else:
        print("\nQuá trình phân cụm không thành công do lỗi.")

