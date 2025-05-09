import pandas as pd
import numpy as np
import os
import traceback
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Cấu hình đường dẫn ---
def configure_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir_for_results = os.path.dirname(script_dir)
    return {
        'estimation': os.path.join(script_dir, "OUTPUT", "estimation_data.csv"),
        'results': os.path.join(parent_dir_for_results, "problem_1", "results1.csv"),
        'model_artifacts_output': os.path.join(script_dir, "OUTPUT_XGB_PROPOSED"),
        'visualizations_output': os.path.join(script_dir, "visualizations_output") 
    }

def convert_special_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_and_merge_data(paths):
    df_est = pd.read_csv(paths['estimation'])
    df_res = pd.read_csv(paths['results'])
    
    position_col_original_name = None
    for col_name_iter in df_res.columns:
        if 'position' in col_name_iter.lower():
            position_col_original_name = col_name_iter
            break
    
    if position_col_original_name:
        df_res = df_res.rename(columns={position_col_original_name: 'Position'})

    merged = pd.merge(
        df_est, df_res,
        left_on=['Player', 'Team_TransferSite'],
        right_on=['Name', 'Team'],
        how='inner'
    )
    return merged
    
def advanced_feature_engineering(df):
    df['PrimaryPosition'] = df['Position'].astype(str).str.split(',', n=1).str[0].str.strip()
    df['PrimaryPosition'] = df['PrimaryPosition'].replace(['', 'nan', 'NaN'], pd.NA).fillna('Unknown')
    
    df = convert_special_numeric(df, [
        'Skill', 'Potential', 
        'Expected: expected goals (xG)', 
        'Expected: expected Assist Goals (xAG)',
        'TransferValue_EUR_Millions' 
    ])
    
    df['Playing Time: minutes'] = pd.to_numeric(df['Playing Time: minutes'], errors='coerce')
    minutes = df['Playing Time: minutes'].fillna(90).clip(lower=90) 
    metrics = {
        'GoalContrib': ['Performance: goals', 'Performance: assists'],
        'DefensiveActions': ['Tackles: TklW', 'Blocks: Int'], 
        'Progression': ['Progression: PrgC', 'Progression: PrgP']
    }
    for name, cols in metrics.items():
        existing_cols = [col for col in cols if col in df.columns]
        if existing_cols: 
            for metric_col in existing_cols:
                df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
            df[f'{name}_per90'] = df[existing_cols].fillna(0).sum(axis=1) / (minutes / 90 + 1e-6) 
        else:
            df[f'{name}_per90'] = 0.0
    
    
    for col_percent in ['Total: Pass completion (Cmp%)', 'Aerial Duels: Won%']:
        if col_percent in df.columns:
            df[col_percent] = df[col_percent].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False) 
            df[col_percent] = pd.to_numeric(df[col_percent], errors='coerce').fillna(0.0) / 100.0
        else:
            df[col_percent] = 0.0

    df['TransferValue_EUR_Millions'] = pd.to_numeric(df['TransferValue_EUR_Millions'], errors='coerce')
    if (df['TransferValue_EUR_Millions'] < 0).any():
        df['TransferValue_EUR_Millions'] = df['TransferValue_EUR_Millions'].clip(lower=0)
    df['Log_TransferValue'] = np.log1p(df['TransferValue_EUR_Millions'])

    return df

def prepare_model_data(df):
    features_candidates = [
        'Age', 'PrimaryPosition', 'Skill', 'Potential', 
        'GoalContrib_per90', 'DefensiveActions_per90', 'Progression_per90',
        'Expected: expected goals (xG)', 'Expected: expected Assist Goals (xAG)',
        'Total: Pass completion (Cmp%)', 'Aerial Duels: Won%'
    ]
    
    numeric_feature_names = []
    for f_name in features_candidates:
        if f_name in df.columns and f_name != 'PrimaryPosition':
            if pd.api.types.is_numeric_dtype(df[f_name]):
                numeric_feature_names.append(f_name)
            else: 
                try:
                    df[f_name] = pd.to_numeric(df[f_name], errors='coerce')
                    if pd.api.types.is_numeric_dtype(df[f_name]):
                         numeric_feature_names.append(f_name)
                    else:
                        print(f"  Warning (prepare_model_data): Column '{f_name}' could not be converted to numeric and will be excluded.")
                except: 
                     print(f"  Warning (prepare_model_data): Error converting column '{f_name}' to numeric. It will be excluded.")

    categorical_feature_names = ['PrimaryPosition'] 

    numeric_df = df[numeric_feature_names].copy()
    for col in numeric_df.columns: 
        if numeric_df[col].isnull().any():
            numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(numeric_df)

    le_dict = {}
    X_categorical_encoded_list = []
    final_categorical_feature_names_encoded = []

    for cat_col in categorical_feature_names:
        if cat_col in df.columns:
            le = LabelEncoder()
            encoded_col = le.fit_transform(df[cat_col].astype(str)) 
            X_categorical_encoded_list.append(encoded_col.reshape(-1, 1))
            le_dict[cat_col] = le
            final_categorical_feature_names_encoded.append(f"{cat_col}_enc")
        else: 
            print(f"  Warning (prepare_model_data): Categorical feature '{cat_col}' not found, though expected.")

    final_feature_names = []
    if X_categorical_encoded_list:
        X_categorical_encoded = np.concatenate(X_categorical_encoded_list, axis=1)
        if X_numeric_scaled.shape[1] > 0 : 
            X = np.concatenate((X_numeric_scaled, X_categorical_encoded), axis=1)
            final_feature_names = numeric_feature_names + final_categorical_feature_names_encoded
        else: 
            X = X_categorical_encoded
            final_feature_names = final_categorical_feature_names_encoded
    elif X_numeric_scaled.shape[1] > 0 : 
        X = X_numeric_scaled
        final_feature_names = numeric_feature_names
    else: 
        raise ValueError("No features available to form X matrix.")

    if 'Log_TransferValue' not in df.columns:
        raise KeyError("Target variable 'Log_TransferValue' not found.")
    y = df['Log_TransferValue']
    
    if X.shape[0] == 0 or X.shape[0] != len(y):
        raise ValueError(f"Feature matrix X shape {X.shape} is empty or mismatched with y shape {len(y)}.")
        
    return X, y, scaler, le_dict, final_feature_names 

def train_model(X, y): 
    param_grid = {
        'max_depth': [4, 6, 8], 'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5], 'n_estimators': [100, 500, 1000] 
    }
    
    model_xgb = xgb.XGBRegressor(early_stopping_rounds=50, random_state=42)
    try:
        config = xgb.get_config()
        if config.get('USE_CUDA', False) or config.get('gpu_id', -1) != -1:
            model_xgb.set_params(tree_method='gpu_hist')
            print("  Info (train_model): Attempting to use GPU for XGBoost.")
        else:
            model_xgb.set_params(tree_method='hist')
            print("  Info (train_model): Using CPU (hist) for XGBoost (GPU not available or XGBoost not built with CUDA).")
    except Exception as e_gpu:
        model_xgb.set_params(tree_method='hist')
        print(f"  Info (train_model): Could not configure GPU, using CPU (hist). Error: {e_gpu}")

    search = RandomizedSearchCV(
        model_xgb, param_grid, n_iter=20, cv=5, 
        scoring='r2', n_jobs=-1, random_state=42 
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Training model with X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False) 
    print(f"  Best parameters found: {search.best_params_}")
    return search.best_estimator_



if __name__ == "__main__":
    paths = configure_paths()
    try:
        os.makedirs(paths['model_artifacts_output'], exist_ok=True)
        os.makedirs(paths['visualizations_output'], exist_ok=True) 
        print(f"Output for model artifacts will be saved to: {paths['model_artifacts_output']}")
        print(f"Output for visualizations will be saved to: {paths['visualizations_output']}")
        
        print("\n1. Starting data loading and merging...")
        merged_data = load_and_merge_data(paths)
        if merged_data is None or merged_data.empty:
            print("Critical Error: Failed to load/merge data or data is empty. Exiting.")
            exit(1) 
        print(f"   - Merged data shape: {merged_data.shape}")

        print("\n2. Starting advanced feature engineering...")
        processed_data = advanced_feature_engineering(merged_data.copy()) 
        if processed_data is None or processed_data.empty:
            print("Critical Error: Failed to process data or data is empty. Exiting.")
            exit(1)
        print(f"   - Processed data shape: {processed_data.shape}")
        
        if 'Log_TransferValue' not in processed_data.columns:
            print("Critical Error: Target 'Log_TransferValue' not found post-processing. Exiting.")
            exit(1)

        print("\n3. Starting data preparation for model...")
        X, y, scaler, le_dict, feature_names = prepare_model_data(processed_data) 
        print(f"   - Model features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        print(f"   - Feature names for model: {feature_names}")

        print("\n4. Starting model training...")
        model = train_model(X, y) 
        
        print("\n5. Evaluating model...")
        y_pred = model.predict(X) 
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print(f"   - R² score (on full processed data): {r2:.3f}")
        print(f"   - MSE (on full processed data): {mse:.3f}")
        
        # --- BẮT ĐẦU PHẦN TRỰC QUAN HÓA (LƯU VÀO THƯ MỤC visualizations_output) ---
        print("\n6. Generating and saving visualizations...")
        
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['font.family'] = 'sans-serif' 

        # 6.1 Biểu đồ Giá trị Thực tế vs. Giá trị Dự đoán
        plt.figure(figsize=(10, 10))
        plt.scatter(y, y_pred, alpha=0.6, edgecolors='k', s=70, label="Dữ liệu điểm")
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label="Dự đoán hoàn hảo (y=x)")
        plt.xlabel('Giá trị Thực tế (Log_TransferValue)', fontsize=14, labelpad=15) # Thêm labelpad
        plt.ylabel('Giá trị Dự đoán (Log_TransferValue)', fontsize=14, labelpad=15) # Thêm labelpad
        plt.title('Giá trị Thực tế vs. Giá trị Dự đoán', fontsize=16, fontweight='bold')
        plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$\nMSE = {mse:.3f}', 
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(paths['visualizations_output'], 'actual_vs_predicted.png'))
        plt.close() 
        print(f"   - Saved: actual_vs_predicted.png to {paths['visualizations_output']}")

        # 6.2 Biểu đồ Phần dư
        residuals = y - y_pred
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, edgecolor='k', s=70, hue=residuals, palette="coolwarm", legend=False)
        plt.axhline(y=0, color='black', linestyle='--', lw=2)
        plt.xlabel('Giá trị Dự đoán (Log_TransferValue)', fontsize=14, labelpad=15) # Thêm labelpad
        plt.ylabel('Phần dư (Lỗi = Thực tế - Dự đoán)', fontsize=14, labelpad=15) # Thêm labelpad
        plt.title('Biểu đồ Phần dư', fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(paths['visualizations_output'], 'residual_plot.png'))
        plt.close()
        print(f"   - Saved: residual_plot.png to {paths['visualizations_output']}")

        # 6.3 Tầm quan trọng Đặc trưng của XGBoost
        if hasattr(model, 'feature_importances_') and feature_names and len(feature_names) == X.shape[1]:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            plt.figure(figsize=(12, max(8, len(feature_names) * 0.5)))
            sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis', edgecolor='black')
            plt.xlabel("Mức độ quan trọng (Feature Importance Score)", fontsize=14, labelpad=15) # Thêm labelpad
            plt.ylabel("Đặc trưng", fontsize=14, labelpad=15) # Thêm labelpad
            plt.title("Tầm quan trọng của Đặc trưng (XGBoost)", fontsize=16, fontweight='bold')
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            for index, value in enumerate(feature_importance_df['importance']):
                plt.text(value + 0.001, index, f'{value:.3f}', va='center', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(paths['visualizations_output'], 'xgb_feature_importance.png'))
            plt.close()
            print(f"   - Saved: xgb_feature_importance.png to {paths['visualizations_output']}")
        else:
            print("   - Warning: Không thể tạo biểu đồ feature importance của XGBoost.")

        # 6.4 Giải thích bằng SHAP Summary Plot
        print("   - Đang tính toán giá trị SHAP (có thể mất một chút thời gian)...")
        if X.shape[1] == len(feature_names) and len(feature_names) > 0 :
            X_shap_df = pd.DataFrame(X, columns=feature_names)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap_df) 

                plt.figure(figsize=(14, max(8, len(feature_names) * 0.4))) # Đặt kích thước figure trước khi gọi SHAP
                shap.summary_plot(shap_values, X_shap_df, plot_type="dot", show=False, color_bar_label='Giá trị Đặc trưng (Cao/Thấp)')
                fig = plt.gcf() 
                fig.suptitle("SHAP Summary Plot - Ảnh hưởng & Tầm quan trọng của Đặc trưng", fontsize=16, fontweight='bold', y=1.0) 
                plt.subplots_adjust(top=0.9) 
                plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95]) 
                plt.savefig(os.path.join(paths['visualizations_output'], 'shap_summary_plot_dot.png'), bbox_inches='tight')
                plt.close(fig) 
                print(f"   - Saved: shap_summary_plot_dot.png to {paths['visualizations_output']}")
            except Exception as e_shap:
                print(f"   - Lỗi khi tạo SHAP plot: {e_shap}")
                traceback.print_exc()
        else:
            print("   - Warning: Không thể tạo DataFrame cho SHAP. SHAP plots skipped.")
        
        print("   - Visualizations generated and saved.")

        # ... (phần còn lại của mã, ví dụ: lưu model) ...
        print(f"\n7. Saving model and preprocessors to {paths['model_artifacts_output']}...")
        joblib.dump(model, os.path.join(paths['model_artifacts_output'], 'optimized_xgb_model.pkl'))
        if scaler: 
            joblib.dump(scaler, os.path.join(paths['model_artifacts_output'], 'scaler.pkl'))
        if le_dict : 
            joblib.dump(le_dict, os.path.join(paths['model_artifacts_output'], 'label_encoders.pkl'))
        if feature_names: 
            joblib.dump(feature_names, os.path.join(paths['model_artifacts_output'], 'feature_names.pkl')) 
        print(f"   - Model and preprocessors saved.")

    except FileNotFoundError as fnf_error:
        print(f"Lỗi FileNotFoundError trong quá trình thực thi: {fnf_error}")
    except (KeyError, ValueError) as e_data: 
        print(f"Lỗi dữ liệu hoặc xử lý trong quá trình thực thi: {e_data}")
        traceback.print_exc()
    except Exception as e: 
        print(f"Lỗi không mong muốn xảy ra trong quá trình thực thi: {e}")
        traceback.print_exc()