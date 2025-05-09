import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from player_clustering_kmeans import cluster_players_kmeans, ensure_dir

# --- Cấu hình ---
PLAYER_COL = 'Name'
CSV_RELATIVE_PATH = os.path.join('..', 'problem_1', 'results1.csv')
PCA_COMPONENTS = 2

script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder_name = 'pca_kmeans_viz_results'
OUTPUT_DIR_PCA = os.path.join(script_dir, output_folder_name)


# --- Hàm trợ giúp ---
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    
# --- Hàm chính ---
def visualize_pca_clusters(csv_path: str, player_col: str, max_k: int = 10, n_pca_components: int = 2):
    ensure_dir(OUTPUT_DIR_PCA)

    # --- 1. Thực hiện K-Means Clustering (Gọi hàm đã import) ---
    df_clustered_sorted, scaler, features_scaled_df, numeric_cols = cluster_players_kmeans(csv_path, player_col)

    # Lấy số cụm tối ưu đã được chọn trong hàm K-Means từ kết quả trả về
    optimal_k = df_clustered_sorted['Cluster'].nunique()
    print(f"[PCA Viz] Số cụm được sử dụng (từ kết quả K-Means): K = {optimal_k}")

    # --- 2. Áp dụng PCA để giảm chiều ---
    pca = PCA(n_components=n_pca_components)
    features_pca = pca.fit_transform(features_scaled_df) 
   
    pca_df = pd.DataFrame(data=features_pca,
             columns=[f'PC{i+1}' for i in range(n_pca_components)],
             index=df_clustered_sorted.index) 
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"[PCA Viz] Tỷ lệ phương sai được giải thích bởi {n_pca_components} thành phần: {explained_variance:.4f}")

    # --- 3. Kết hợp tọa độ PCA vào DataFrame đã phân cụm ---
    df_pca_clustered = pd.concat([df_clustered_sorted, pca_df], axis=1)

    # --- 4. Trực quan hóa 2D bằng Scatter Plot ---
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("hsv", optimal_k) 

    scatter_plot = sns.scatterplot(
        x=f'PC1',
        y=f'PC2',
        hue='Cluster',
        palette=palette, 
        data=df_pca_clustered,
        legend='full', 
        alpha=0.7 
    )
    plt.title(f'Phân cụm cầu thủ K-Means (K={optimal_k}) - Trực quan hóa PCA 2D')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.legend(title='Cluster')
    pca_plot_path = os.path.join(OUTPUT_DIR_PCA, f'kmeans_pca_{optimal_k}_clusters_viz.png')
    plt.savefig(pca_plot_path)
    plt.close() 

    # --- 5. Lưu kết quả cuối cùng (tùy chọn) ---
    output_csv_path = os.path.join(OUTPUT_DIR_PCA, f'player_clusters_with_pca_{optimal_k}.csv')
    df_pca_clustered.to_csv(output_csv_path, index=False)
    print(f"[PCA Viz] Đã lưu kết quả phân cụm kèm tọa độ PCA vào: {output_csv_path}")
    return df_pca_clustered 


if __name__ == '__main__':
    final_data_with_pca = visualize_pca_clusters(CSV_RELATIVE_PATH, PLAYER_COL, PCA_COMPONENTS)

    if final_data_with_pca is not None:
        print("\n[PCA Viz] Ví dụ một vài cầu thủ (bao gồm tọa độ PCA):")

        numeric_cols_in_final = final_data_with_pca.select_dtypes(include=np.number).columns.drop(['Cluster','PC1','PC2'], errors='ignore').tolist()
        for cluster_id in sorted(final_data_with_pca['Cluster'].unique()):
            print(f"\n--- Cụm {cluster_id} ---")
            sample_players = final_data_with_pca[final_data_with_pca['Cluster'] == cluster_id].head(5)
            cols_to_show = [PLAYER_COL] + numeric_cols_in_final[:3] + ['Cluster', 'PC1', 'PC2']
            cols_to_show = [col for col in cols_to_show if col in sample_players.columns]
            print(sample_players[cols_to_show].round(2)) 