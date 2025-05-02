import pandas as pd
import numpy as np
import re
import os # Import thư viện os

# Import necessary sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Step 1: Load Data ---
print("Bước 1: Tải dữ liệu...")

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "OUTPUT")

# Xây dựng đường dẫn đến thư mục problem_1 (nằm cùng cấp với script_dir)
parent_dir = os.path.dirname(script_dir) # Lấy thư mục cha (Assignment_1)
problem1_dir = os.path.join(parent_dir, "problem_1")

# Tạo đường dẫn đầy đủ đến các tệp CSV
estimation_file_path = os.path.join(output_dir, "estimation_data.csv")
# results.csv nằm trong thư mục problem_1
results_file_path = os.path.join(problem1_dir, "results.csv")
# ---------------------------------

try:
    # Tải dữ liệu bằng đường dẫn đã xác định
    df_estimation = pd.read_csv(estimation_file_path)
    df_results = pd.read_csv(results_file_path)
    print(f"Tải dữ liệu thành công từ '{estimation_file_path}' và '{results_file_path}'.")
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy tệp tin. Vui lòng kiểm tra lại đường dẫn và cấu trúc thư mục.")
    print(f"Đường dẫn đang tìm kiếm: '{os.path.abspath(estimation_file_path)}', '{os.path.abspath(results_file_path)}'")
    print(f"Thư mục gốc của script (hoặc thư mục làm việc): '{script_dir}'")
    exit() # Exit if files not found
except Exception as e:
    print(f"Đã xảy ra lỗi khi đọc tệp: {e}")
    exit()

# --- Step 2: Prepare and Merge Data ---
print("\nBước 2: Chuẩn bị và kết hợp dữ liệu...")
try:
    # --- 2a. Prepare df_estimation ---
    df_model_base = df_estimation[['Player', 'Age', 'Position', 'Skill', 'Potential', 'TransferValue_EUR_Millions']].copy()
    df_model_base.rename(columns={'TransferValue_EUR_Millions': 'TargetValue'}, inplace=True)
    df_model_base['Age'] = df_model_base['Age'].astype(int)

    # --- 2b. Prepare df_results ---
    cols_to_select = {
        'Name': 'Player', # Key for merging
        'Playing Time: minutes': 'Minutes',
        'Performance: goals': 'Goals',
        'Performance: assists': 'Assists',
        'Expected: expected goals (xG)': 'xG',
        'Expected: expected Assist Goals (xAG)': 'xAG',
        'Tackles: TklW': 'TacklesWon',
        'Blocks: Int': 'Interceptions',
        'Total: Pass completion (Cmp%)': 'PassAcc%',
        'Aerial Duels: Won%': 'AerialWon%',
        'Progression: PrgC': 'ProgProgCarries',
        'Progression: PrgP': 'ProgPasses'
    }
    existing_cols_results = [col for col in cols_to_select.keys() if col in df_results.columns]
    if len(existing_cols_results) < len(cols_to_select):
        missing = set(cols_to_select.keys()) - set(existing_cols_results)
        print(f"Cảnh báo: Các cột sau không tìm thấy trong results.csv: {missing}")
    df_performance = df_results[existing_cols_results].copy()
    rename_map = {k:v for k,v in cols_to_select.items() if k in existing_cols_results}
    df_performance.rename(columns=rename_map, inplace=True)


    # --- 2c. Clean selected performance columns ---
    def clean_percentage(x):
        if isinstance(x, str):
            try:
                return float(x.replace('%', '').strip()) / 100.0
            except ValueError:
                return np.nan
        elif isinstance(x, (int, float)):
            return x / 100.0 if x > 1.0 else x
        return np.nan

    percent_cols = [col for col in ['PassAcc%', 'AerialWon%'] if col in df_performance.columns]
    numeric_cols = [col for col in ['Minutes', 'Goals', 'Assists', 'xG', 'xAG', 'TacklesWon', 'Interceptions', 'ProgProgCarries', 'ProgPasses'] if col in df_performance.columns]

    for col in percent_cols:
        df_performance[col] = df_performance[col].apply(clean_percentage)

    for col in numeric_cols:
        df_performance[col] = pd.to_numeric(df_performance[col], errors='coerce')

    for col in percent_cols + numeric_cols:
         if df_performance[col].isnull().any():
            median_val = df_performance[col].median()
            df_performance[col].fillna(median_val, inplace=True)

    # --- 2d. Merge DataFrames ---
    if 'Player' not in df_performance.columns:
        print("Lỗi: Cột 'Player' không tồn tại trong dữ liệu hiệu suất. Không thể kết hợp.")
        exit()

    df_merged = pd.merge(df_model_base, df_performance, on='Player', how='left')

    stats_cols_to_fill = df_performance.columns.drop('Player')
    fill_values = {}
    for col in stats_cols_to_fill:
        if col in df_merged.columns:
             if df_merged[col].isnull().any():
                median_val = df_merged[col].median()
                fill_values[col] = median_val
                df_merged[col].fillna(median_val, inplace=True)
                print(f"Đã gán giá trị trung vị ({median_val:.3f}) cho '{col}' cho các cầu thủ bị thiếu sau khi merge.")

    print("Chuẩn bị và kết hợp dữ liệu hoàn tất.")
    print(f"Kích thước dữ liệu kết hợp: {df_merged.shape}")
    if df_merged.isnull().sum().sum() > 0:
        print("Cảnh báo: Vẫn còn giá trị thiếu!")
        print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])
    else:
        print("Không còn giá trị thiếu trong dữ liệu kết hợp.")

except Exception as e:
    print(f"Đã xảy ra lỗi trong Bước 2: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- Step 3: EDA & Preprocessing ---
print("\nBước 3: EDA & Tiền xử lý...")
try:
    # --- 3a. Log Transform Target ---
    if (df_merged['TargetValue'] <= 0).any():
        print("Cảnh báo: TargetValue <= 0. Sử dụng log1p.")
        df_merged['TargetValue_log'] = np.log1p(df_merged['TargetValue'])
    else:
        df_merged['TargetValue_log'] = np.log(df_merged['TargetValue'])
    target_variable = 'TargetValue_log'

    # --- 3b. Define Features (X) and Target (y) ---
    X = df_merged.drop(['Player', 'TargetValue', target_variable], axis=1, errors='ignore')
    y = df_merged[target_variable]

    # --- 3c. Identify Feature Types ---
    categorical_features = ['Position'] if 'Position' in X.columns else []
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # --- 3d. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test_original = np.expm1(y_test) if target_variable == 'TargetValue_log' else np.exp(y_test)

    # --- 3e. Create Preprocessing Pipeline ---
    transformers_list = []
    if numerical_features:
        transformers_list.append(('num', StandardScaler(), numerical_features))
    if categorical_features:
        transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

    if not transformers_list:
        print("Lỗi: Không có đặc trưng để tiền xử lý.")
        exit()

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')

    # --- 3f. Apply Preprocessing ---
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    all_feature_names = []
    if numerical_features:
        all_feature_names.extend(numerical_features)
    if categorical_features:
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            all_feature_names.extend(list(ohe_feature_names))
        except Exception as e:
            num_numeric = len(numerical_features) if numerical_features else 0
            num_ohe = X_train_processed.shape[1] - num_numeric
            all_feature_names.extend([f'cat_{i}' for i in range(num_ohe)])

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)

    print("Tiền xử lý hoàn tất.")

except Exception as e:
    print(f"Đã xảy ra lỗi trong Bước 3: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- Step 4: Model Training & Evaluation ---
print("\nBước 4: Huấn luyện và Đánh giá Mô hình...")
try:
    ridge_model = Ridge(alpha=1.0, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)

    models = {"Ridge Regression": ridge_model, "Random Forest": rf_model}
    results = {}

    for name, model in models.items():
        print(f"  Huấn luyện mô hình: {name}...")
        model.fit(X_train_processed_df, y_train)
        print(f"  Dự đoán với mô hình: {name}...")
        y_pred_log = model.predict(X_test_processed_df)
        y_pred_original = np.expm1(y_pred_log) if target_variable == 'TargetValue_log' else np.exp(y_pred_log)
        y_pred_original[y_pred_original < 0] = 0

        print(f"  Đánh giá mô hình: {name}...")
        r2 = r2_score(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        print(f"  Kết quả {name}: R2={r2:.4f}, MAE={mae:.4f}M EUR, RMSE={rmse:.4f}M EUR")

except Exception as e:
    print(f"Đã xảy ra lỗi trong Bước 4: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- Step 5: Display Results & Feature Importance ---
print("\nBước 5: Hiển thị Kết quả...")
try:
    if results:
        results_df = pd.DataFrame(results).T
        results_df.index.name = 'Model'
        print("\n--- So sánh Kết quả Đánh giá Mô hình ---")
        print(results_df.to_markdown(numalign="left", stralign="left"))
    else:
        print("\nKhông có kết quả đánh giá để hiển thị.")

    if "Random Forest" in models and "Random Forest" in results and all_feature_names:
        rf_importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)
        rf_importances = rf_importances.sort_values(ascending=False)
        print("\n--- Mức độ quan trọng của Đặc trưng (Random Forest - Top 20) ---")
        print(rf_importances.head(20).to_markdown(numalign="left", stralign="left"))

except Exception as e:
    print(f"Đã xảy ra lỗi trong Bước 5: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Hoàn thành ---")