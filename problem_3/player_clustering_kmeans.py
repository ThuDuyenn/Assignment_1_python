import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình ---
PLAYER_COL = 'Name'
CSV_RELATIVE_PATH = os.path.join('..', 'problem_1', 'results.csv')
MAX_K_TO_TEST = 15

script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder_name = 'kmeans_analysis_results'
OUTPUT_DIR = os.path.join(script_dir, output_folder_name)

# --- Hàm trợ giúp ---
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# --- Hàm chính ---
def cluster_players_kmeans(csv_path: str, player_col: str, max_k: int = 10):
    # --- 1. Nạp dữ liệu ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_csv_path = os.path.join(current_script_dir, csv_path)
    df = pd.read_csv(absolute_csv_path)

    # --- 2. Chuẩn bị dữ liệu ---
    # Chọn các cột số để phân cụm
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features_df = df[numeric_cols].copy()

    # Xử lý giá trị thiếu (NaN) bằng cách thay thế bằng giá trị trung bình của cột
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features_df)
    features_imputed_df = pd.DataFrame(features_imputed, columns=numeric_cols, index=df.index)

    # --- 3. Chuẩn hóa dữ liệu ---
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=numeric_cols, index=df.index)

    # --- 4. Xác định số cụm tối ưu (K) bằng phương pháp Elbow ---
    wcss = [] # Within-Cluster Sum of Squares
    possible_k_values = range(1, max_k + 1)

    for k in possible_k_values:
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
    plt.savefig(elbow_plot_path)
    print(f"Đã lưu biểu đồ Elbow vào: {elbow_plot_path}")
    print("=> Hãy xem biểu đồ Elbow và chọn giá trị K tại điểm 'khuỷu tay'.")
    plt.close()

    # --- Tạm dừng để nhập K tối ưu ---
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
    
    final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init='auto', random_state=42)
    final_kmeans.fit(features_scaled_df)

    cluster_labels = final_kmeans.labels_

    # --- 6. Thêm nhãn cụm vào DataFrame gốc ---
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels

    # --- 7. Phân tích và Lưu kết quả hiệu quả hơn ---
    print("\n--- Phân tích sơ bộ các cụm ---")
    # Tính tâm cụm ở thang đo gốc
    cluster_centroids_original_scale = scaler.inverse_transform(final_kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(cluster_centroids_original_scale, columns=numeric_cols)
    centroid_df.index.name = 'Cluster'
    print("Giá trị trung bình của các đặc trưng số cho mỗi cụm (thang đo gốc):")
    print(centroid_df.round(2))
    centroid_output_path = os.path.join(OUTPUT_DIR, 'cluster_centroids.csv')
    centroid_df.round(2).to_csv(centroid_output_path, index=True)
    print(f"\nĐã lưu bảng phân tích tâm cụm vào: {centroid_output_path}")
    
    # Đếm số lượng cầu thủ trong mỗi cụm
    print("\nSố lượng cầu thủ trong mỗi cụm:")
    print(df_clustered['Cluster'].value_counts().sort_index())
    print("---------------------------------")

    df_clustered_sorted = df_clustered.sort_values(by='Cluster').reset_index(drop=True)

    # Lưu kết quả đã sắp xếp
    output_csv_path = os.path.join(OUTPUT_DIR, 'player_clusters_sorted.csv') 
    df_clustered_sorted.to_csv(output_csv_path, index=False)
    print(f"Đã lưu kết quả phân cụm (đã sắp xếp) vào: {output_csv_path}")

    return df_clustered, scaler, features_scaled_df, numeric_cols


# --- Main Execution ---
if __name__ == '__main__':
    clustered_data, _, _, numeric_cols_list = cluster_players_kmeans(CSV_RELATIVE_PATH, PLAYER_COL, MAX_K_TO_TEST)

    if clustered_data is not None:
        print("\nVí dụ một vài cầu thủ từ mỗi cụm:")
        if numeric_cols_list:
            for cluster_id in sorted(clustered_data['Cluster'].unique()):
                print(f"\n--- Cụm {cluster_id} ---")
                sample_players = clustered_data[clustered_data['Cluster'] == cluster_id].head(5)
                cols_to_show = [PLAYER_COL] + numeric_cols_list[:5] + ['Cluster']
                cols_to_show = [col for col in cols_to_show if col in sample_players.columns]
                if len(cols_to_show) > 2:
                    print(sample_players[cols_to_show])
                else:
                    print(sample_players[[PLAYER_COL, 'Cluster']])