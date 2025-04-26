import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA # Thêm thư viện PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình ---
PLAYER_COL = 'Name'
CSV_RELATIVE_PATH = os.path.join('..', 'problem_1', 'results.csv')
MAX_K_TO_TEST = 10
OUTPUT_DIR = 'kmeans_analysis'
# Số chiều sau khi giảm bằng PCA (cho mục đích trực quan hóa)
PCA_COMPONENTS = 2

# --- Hàm trợ giúp ---
def ensure_dir(path: str):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)

# --- Hàm chính ---
def cluster_players_kmeans_pca_viz(csv_path: str, player_col: str, max_k: int = 10, n_pca_components: int = 2):
    """
    Thực hiện phân cụm K-Means, giảm chiều bằng PCA và trực quan hóa 2D.

    Args:
        csv_path: Đường dẫn đến file CSV.
        player_col: Tên cột chứa định danh cầu thủ.
        max_k: Số cụm tối đa để kiểm tra bằng phương pháp Elbow.
        n_pca_components: Số thành phần chính để giảm chiều (mặc định là 2).

    Returns:
        DataFrame chứa thông tin cầu thủ, nhãn cụm và tọa độ PCA,
        hoặc None nếu có lỗi.
    """
    try:
        # --- 1. Nạp dữ liệu ---
        # (Giữ nguyên như trước)
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
            print(f"Lỗi: Không tìm thấy cột cầu thủ '{player_col}' trong CSV.")
            return None

        # --- 2. Chuẩn bị dữ liệu ---
        # (Giữ nguyên như trước)
        print("\nĐang chuẩn bị dữ liệu...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
             print("Lỗi: Không tìm thấy cột số nào để thực hiện phân cụm.")
             return None
        print(f"Các cột số được chọn: {', '.join(numeric_cols)}")
        features_df = df[numeric_cols].copy()
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features_df)
        features_imputed_df = pd.DataFrame(features_imputed, columns=numeric_cols, index=df.index)
        print(f"Đã xử lý giá trị thiếu.")

        # --- 3. Chuẩn hóa dữ liệu ---
        # (Giữ nguyên như trước)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed_df)
        features_scaled_df = pd.DataFrame(features_scaled, columns=numeric_cols, index=df.index)
        print("Đã chuẩn hóa dữ liệu.")

        # --- 4. Xác định số cụm tối ưu (K) bằng phương pháp Elbow ---
        # (Giữ nguyên như trước, chạy K-Means trên dữ liệu đã chuẩn hóa *đầy đủ*)
        print(f"\nĐang tìm số cụm tối ưu (K) bằng phương pháp Elbow (tối đa K={max_k})...")
        wcss = []
        possible_k_values = range(1, max_k + 1)
        for k in possible_k_values:
            kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
            kmeans_model.fit(features_scaled_df) # Fit trên dữ liệu đầy đủ chiều đã scale
            wcss.append(kmeans_model.inertia_)

        # Vẽ biểu đồ Elbow (Giữ nguyên)
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

        # Nhập K tối ưu (Giữ nguyên)
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
        # (Giữ nguyên như trước, chạy trên dữ liệu đầy đủ chiều đã scale)
        print(f"\nĐang áp dụng K-Means với K = {optimal_k}...")
        final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init='auto', random_state=42)
        # Fit lại trên dữ liệu đã scale đầy đủ để lấy nhãn cụm chính xác
        final_kmeans.fit(features_scaled_df)
        cluster_labels = final_kmeans.labels_
        print("Đã gán nhãn cụm cho các cầu thủ.")

        # --- 6. Áp dụng PCA để giảm chiều xuống còn 2 ---
        print(f"\nĐang áp dụng PCA để giảm chiều dữ liệu xuống còn {n_pca_components} thành phần...")
        pca = PCA(n_components=n_pca_components)
        # Fit và transform trên dữ liệu đã scale đầy đủ
        features_pca = pca.fit_transform(features_scaled_df)
        pca_df = pd.DataFrame(data=features_pca,
                              columns=[f'PC{i+1}' for i in range(n_pca_components)],
                              index=df.index)
        print(f"Tỷ lệ phương sai được giải thích bởi {n_pca_components} thành phần chính: {pca.explained_variance_ratio_.sum():.4f}")

        # --- 7. Thêm nhãn cụm và tọa độ PCA vào DataFrame gốc ---
        df_clustered = df.copy()
        df_clustered['Cluster'] = cluster_labels
        # Nối các cột PCA vào df_clustered
        df_clustered = pd.concat([df_clustered, pca_df], axis=1)
        print("Đã thêm nhãn cụm và tọa độ PCA vào DataFrame.")

        # --- 8. Trực quan hóa 2D bằng Scatter Plot ---
        print("\nĐang tạo biểu đồ phân tán 2D...")
        plt.figure(figsize=(12, 8))
        # Sử dụng seaborn để vẽ đẹp hơn và có legend tự động
        scatter_plot = sns.scatterplot(
            x='PC1',  # Trục X là thành phần chính 1
            y='PC2',  # Trục Y là thành phần chính 2
            hue='Cluster', # Màu sắc dựa trên nhãn cụm
            palette=sns.color_palette("hsv", optimal_k), # Chọn bảng màu
            data=df_clustered,
            legend='full',
            alpha=0.7 # Độ trong suốt của điểm
        )
        plt.title(f'Phân cụm cầu thủ K-Means (K={optimal_k}) - Trực quan hóa PCA 2D')
        plt.xlabel('Thành phần chính 1 (PC1)')
        plt.ylabel('Thành phần chính 2 (PC2)')
        plt.grid(True)

        # Lưu biểu đồ scatter plot
        pca_plot_path = os.path.join(OUTPUT_DIR, f'kmeans_pca_{optimal_k}_clusters.png')
        try:
            plt.savefig(pca_plot_path)
            print(f"Đã lưu biểu đồ PCA 2D vào: {pca_plot_path}")
        except Exception as e:
            print(f"Lỗi khi lưu biểu đồ PCA: {e}")
        finally:
            plt.close() # Đóng plot để giải phóng bộ nhớ

        # --- 9. Phân tích kết quả (Giữ nguyên phần centroid) ---
        print("\n--- Phân tích sơ bộ các cụm ---")
        # Tính giá trị trung bình của các chỉ số cho từng cụm (dựa trên tâm cụm K-Means gốc)
        cluster_centroids_original_scale = scaler.inverse_transform(final_kmeans.cluster_centers_)
        centroid_df = pd.DataFrame(cluster_centroids_original_scale, columns=features_imputed_df.columns)
        centroid_df.index.name = 'Cluster'
        print("Giá trị trung bình của các chỉ số gốc cho mỗi cụm:")
        try:
            from IPython.display import display
            display(centroid_df.round(2))
        except ImportError:
             print(centroid_df.round(2))

        print("\nSố lượng cầu thủ trong mỗi cụm:")
        print(df_clustered['Cluster'].value_counts().sort_index())
        print("---------------------------------")

        # Lưu kết quả cuối cùng (bao gồm cả PCA)
        output_csv_path = os.path.join(OUTPUT_DIR, f'player_clusters_pca_{optimal_k}.csv')
        try:
             df_clustered.to_csv(output_csv_path, index=False)
             print(f"Đã lưu kết quả phân cụm và PCA vào: {output_csv_path}")
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
    print("Bắt đầu quá trình phân cụm K-Means và trực quan hóa PCA...")
    # Gọi hàm mới
    clustered_data = cluster_players_kmeans_pca_viz(CSV_RELATIVE_PATH, PLAYER_COL, MAX_K_TO_TEST, PCA_COMPONENTS)

    if clustered_data is not None:
        print("\nQuá trình phân cụm và trực quan hóa hoàn tất.")
        # Hiển thị ví dụ vẫn giữ nguyên, nhưng giờ df có thêm cột PC1, PC2
        print("\nVí dụ một vài cầu thủ từ mỗi cụm (bao gồm tọa độ PCA):")
        numeric_cols_used = clustered_data.select_dtypes(include=np.number).columns.drop(['Cluster','PC1','PC2'], errors='ignore').tolist()
        for cluster_id in sorted(clustered_data['Cluster'].unique()):
             print(f"\n--- Cụm {cluster_id} ---")
             sample_players = clustered_data[clustered_data['Cluster'] == cluster_id].head(5)
             # Hiển thị tên, vài chỉ số đầu, Cluster, PC1, PC2
             cols_to_show = [PLAYER_COL] + numeric_cols_used[:3] + ['Cluster', 'PC1', 'PC2']
             cols_to_show = [col for col in cols_to_show if col in sample_players.columns]
             print(sample_players[cols_to_show].round(2)) # Làm tròn tọa độ PCA
    else:
        print("\nQuá trình không thành công do lỗi.")