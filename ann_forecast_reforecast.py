# %%
import os
import pandas as pd

# %%
path = './dataset'
datasets_path = path + '/dataset/solar_stations'
# %%
files = [os.path.join(path + '/lag_added_dataset', file) for file in os.listdir(path + '/lag_added_dataset') if
         file.endswith('.csv')]
origin_df_list = []
files.sort()
for file in files:
    print(file)
    df = pd.read_csv(file)

    df = df[[col for col in df.columns if col != 'Direct normal irradiance (W/m2)']]
    df = df[[col for col in df.columns if col != 'Global horizontal irradiance (W/m2)']]
    df = df[[col for col in df.columns if col != 'day']]
    df = df[[col for col in df.columns if col != 'day_sin']]
    df = df[[col for col in df.columns if col != 'day_cos']]
    df = df[[col for col in df.columns if col != 'hour']]
    df = df[[col for col in df.columns if col != 'month']]
    df = df[[col for col in df.columns if col != 'Time']]
    df = df[[col for col in df.columns if col != 'Unnamed: 0.1']]
    df = df[[col for col in df.columns if col != 'Unnamed: 0']]
    origin_df_list.append(df)

print(len(origin_df_list))

# %%
origin_df_list[0].columns
# %%
origin_df_list[0].head()
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math

test_size = 2 / 24
features = ['Total solar irradiance (W/m2)',
            'Air temperature  (°C) ', 'Atmosphere (hpa)', 'Relative humidity (%)',
            'Power (MW)', 'hour_sin', 'hour_cos', 'month_sin',
            'month_cos', 'same_time_mean', 'same_time_std', 'same_time_max',
            'same_time_min', 'same_weekday_mean', 'same_weekday_std',
            'same_weekday_max', 'same_weekday_min', 'overall_mean', 'overall_std',
            'overall_max', 'overall_min', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
            'lag_5', 'lag_6', 'lag_7', 'lag_8', 'diff_15min', 'diff_30min',
            'diff_45min', 'diff_60min', 'diff_75min', 'diff_90min', 'diff_105min']

target = 'Power (MW)'
shifted_column = 'power_shifted'
# %%
import joblib
import numpy as np
from datetime import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb


def do_base_forecast(origin_df_list, site_names):
    forecasted_df_list = []

    for i in range(len(origin_df_list)):
        print("===============================================================")
        print(site_names[i])
        df = origin_df_list[i].copy()

        # 다음 시점 예측을 위한 타겟 시프트
        df[shifted_column] = df[target].shift(-1)
        df = df.dropna().reset_index(drop=True)

        features = [col for col in features if col in df.columns]
        X = df[features]
        y = df[shifted_column]
        time_col = df['Time(year-month-day h:m:s)']

        # 시계열 순서 기반 분할
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        time_train, time_test = time_col.iloc[:split_index], time_col.iloc[split_index:]

        # 정규화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 학습
        print("MLR")
        mlr = LinearRegression()
        mlr.fit(X_train_scaled, y_train)

        print("SVR")
        svr = SVR()
        svr.fit(X_train_scaled, y_train)

        print("LightGBM")
        lgb_model = lgb.LGBMRegressor()
        lgb_model.fit(X_train, y_train)

        print("MLP")
        mlp = MLPRegressor(random_state=42, max_iter=1000)
        mlp.fit(X_train_scaled, y_train)

        # 예측 결과 저장
        df_result = X.copy()
        df_result.loc[:split_index - 1, 'pred_mlr'] = mlr.predict(X_train_scaled)
        df_result.loc[split_index:, 'pred_mlr'] = mlr.predict(X_test_scaled)

        df_result.loc[:split_index - 1, 'pred_svr'] = svr.predict(X_train_scaled)
        df_result.loc[split_index:, 'pred_svr'] = svr.predict(X_test_scaled)

        df_result.loc[:split_index - 1, 'pred_lgb'] = lgb_model.predict(X_train)
        df_result.loc[split_index:, 'pred_lgb'] = lgb_model.predict(X_test)

        df_result.loc[:split_index - 1, 'pred_mlp'] = mlp.predict(X_train_scaled)
        df_result.loc[split_index:, 'pred_mlp'] = mlp.predict(X_test_scaled)

        # 예측값 t+1에 위치시키기
        for col in ['pred_mlr', 'pred_svr', 'pred_lgb', 'pred_mlp']:
            df_result[col] = df_result[col].shift(1)

        # 시간 컬럼 복원
        X_test_result = df_result.copy()
        X_test_result.loc[:split_index - 1, 'Time'] = time_train.values
        X_test_result.loc[split_index:, 'Time'] = time_test.values
        X_test_result.dropna(subset=['pred_mlr', 'pred_svr', 'pred_lgb', 'pred_mlp'], inplace=True)

        forecasted_df_list.append(X_test_result)

        # 모델 저장
        joblib.dump(mlr, f"{path}/result_of_paper/ann/mlr_forecasting_model_{str(site_names[i])}.joblib")
        joblib.dump(svr, f"{path}/result_of_paper/ann/svr_forecasting_model_{str(site_names[i])}.joblib")
        joblib.dump(mlp, f"{path}/result_of_paper/ann/mlp_forecasting_model_{str(site_names[i])}.joblib")
        lgb_model.booster_.save_model(f"{path}/result_of_paper/ann/lgb_forecasting_model_{str(site_names[i])}.txt")
    # 오차 계산

    result_for_reforecast_df_list = []
    for forecasted_df in forecasted_df_list:
        for model in ['mlr', 'svr', 'lgb', 'mlp']:
            forecasted_df[f'pred_{model}'] = forecasted_df[f'pred_{model}'].apply(lambda x: max(x, 0))
            forecasted_df[f'error_{model}'] = forecasted_df.apply(
                lambda row: (row[f'pred_{model}'] - row['Power (MW)']),
                axis=1
            )

        result_for_reforecast_df_list.append(forecasted_df)

    return result_for_reforecast_df_list


selected_df_list = origin_df_list.copy  # select df which you want to forecast. e.g. origin_df_list[0:1]
site_names = [1, 2, 4, 5, 6, 7, 8]  # set selected df's site name e.g. ['first']
operation_hours_array = [
    ('06:00', '21:30'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('06:00', '21:00'),
    ('06:00', '21:00'),
    ('06:00', '19:00')
]  # set operation hours for each selected df


# %%
def save_forecast_result(result_for_reforecast_df_list, site_names):
    for final_result_df, df_index in zip(result_for_reforecast_df_list, site_names):
        forecast_df_list.append(pd.read_csv(f"{path}/result_of_paper/ann/forecast_result_{df_index}_.csv"))


def filter_by_operation_hours(result_for_reforecast_df_list, operation_hours_array):
    filtered_df_list = []

    for df, (start_str, end_str) in zip(result_for_reforecast_df_list, operation_hours_array):
        split_index = int(len(df) * (1 - test_size))
        df_copy = df.iloc[split_index:].copy()
        df_copy['Time'] = pd.to_datetime(df_copy['Time'])

        start_time = time.fromisoformat(start_str)
        end_time = time.fromisoformat(end_str)

        df_copy = df_copy[df_copy['Time'].dt.time.between(start_time, end_time)]
        filtered_df_list.append(df_copy)

    return filtered_df_list


def print_out_forecast_eval_result(filtered_df_list, site_names):
    for result_df, df_index in zip(filtered_df_list, site_names):
        print(
            '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(df_index)

        for forecast_model in ['mlr', 'svr', 'lgb', 'mlp']:
            print(f'reforecast model : {forecast_model.upper()}')

            mse = (result_df[f'error_{forecast_model}'] ** 2).mean()
            mae = result_df[f'error_{forecast_model}'].abs().mean()
            rmse = np.sqrt(mse)

            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")


result_for_reforecast_df_list = do_base_forecast(origin_df_list, site_names)
save_forecast_result(result_for_reforecast_df_list, site_names)
filtered_df_list = filter_by_operation_hours(result_for_reforecast_df_list, operation_hours_array)
print_out_forecast_eval_result(filtered_df_list, site_names)
# %% md
# ## Reforecast
# %%
features = ['Total solar irradiance (W/m2)',
            'Air temperature  (°C) ', 'Atmosphere (hpa)', 'Relative humidity (%)',
            'Power (MW)', 'hour_sin', 'hour_cos', 'month_sin',
            'month_cos', 'same_time_mean', 'same_time_std', 'same_time_max',
            'same_time_min', 'same_weekday_mean', 'same_weekday_std',
            'same_weekday_max', 'same_weekday_min', 'overall_mean', 'overall_std',
            'overall_max', 'overall_min', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
            'lag_5', 'lag_6', 'lag_7', 'lag_8', 'diff_15min', 'diff_30min',
            'diff_45min', 'diff_60min', 'diff_75min', 'diff_90min', 'diff_105min']

# 예측 대상 오차 (error) 컬럼명
mlr_target = 'error_mlr'
svr_target = 'error_svr'
lgb_target = 'error_lgb'
mlp_target = 'error_mlp'
targets = [mlr_target, svr_target, lgb_target, mlp_target]

# 각 모델별 재예측에 사용될 feature 목록
forecast_mlr_features = features + ['error_mlr', 'pred_mlr', 'same_time_mean_error_mlr',
                                    'same_time_std_error_mlr',
                                    'same_time_max_error_mlr',
                                    'same_time_min_error_mlr',
                                    'same_weekday_mean_error_mlr',
                                    'same_weekday_std_error_mlr',
                                    'same_weekday_max_error_mlr',
                                    'same_weekday_min_error_mlr',
                                    'overall_mean_error_mlr',
                                    'overall_std_error_mlr',
                                    'overall_max_error_mlr',
                                    'overall_min_error_mlr']

forecast_svr_features = features + ['error_svr', 'pred_svr', 'same_time_mean_error_svr',
                                    'same_time_std_error_svr',
                                    'same_time_max_error_svr',
                                    'same_time_min_error_svr',
                                    'same_weekday_mean_error_svr',
                                    'same_weekday_std_error_svr',
                                    'same_weekday_max_error_svr',
                                    'same_weekday_min_error_svr',
                                    'overall_mean_error_svr',
                                    'overall_std_error_svr',
                                    'overall_max_error_svr',
                                    'overall_min_error_svr']

forecast_lgb_features = features + ['error_lgb', 'pred_lgb', 'same_time_mean_error_lgb',
                                    'same_time_std_error_lgb',
                                    'same_time_max_error_lgb',
                                    'same_time_min_error_lgb',
                                    'same_weekday_mean_error_lgb',
                                    'same_weekday_std_error_lgb',
                                    'same_weekday_max_error_lgb',
                                    'same_weekday_min_error_lgb',
                                    'overall_mean_error_lgb',
                                    'overall_std_error_lgb',
                                    'overall_max_error_lgb',
                                    'overall_min_error_lgb']

forecast_mlp_features = features + ['error_mlp', 'pred_mlp', 'same_time_mean_error_mlp',
                                    'same_time_std_error_mlp',
                                    'same_time_max_error_mlp',
                                    'same_time_min_error_mlp',
                                    'same_weekday_mean_error_mlp',
                                    'same_weekday_std_error_mlp',
                                    'same_weekday_max_error_mlp',
                                    'same_weekday_min_error_mlp',
                                    'overall_mean_error_mlp',
                                    'overall_std_error_mlp',
                                    'overall_max_error_mlp',
                                    'overall_min_error_mlp']

# 모델별 feature list
features_list = [forecast_mlr_features, forecast_svr_features, forecast_lgb_features, forecast_mlp_features]

# shift된 오차 타겟명
shifted_mlr_target = 'error_mlr_shifted'
shifted_svr_target = 'error_svr_shifted'
shifted_lgb_target = 'error_lgb_shifted'
shifted_mlp_target = 'error_mlp_shifted'
shifted_targets = [shifted_mlr_target, shifted_svr_target, shifted_lgb_target, shifted_mlp_target]

# %%
import joblib
import numpy as np
from datetime import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


def do_reforecast(targets, shifted_targets, features_list, site_names, result_for_reforecast_df_list):
    final_result_lists = [[], [], [], []]

    for forecast_model, target, shifted_target, final_result_list, features in zip(
            ['mlr', 'svr', 'lgb', 'mlp'],
            targets,
            shifted_targets,
            final_result_lists,
            features_list):

        print('============================================================================================')
        print(f'forecast model : {forecast_model.upper()}')

        reforecasted_df_list = []
        for df, df_index in zip(result_for_reforecast_df_list, site_names):
            print('----------------------------------------------------------------------------------------------')
            print(df_index)

            df[shifted_target] = df[target].shift(-1)
            df = df.dropna().reset_index(drop=True)

            features = [col for col in features if col in df.columns]

            X = df[features]
            y = df[shifted_target]
            time_col = df['Time']

            # 시계열 순서 기반 분할
            split_index = int(len(X) * (1 - test_size))

            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            time_train, time_test = time_col.iloc[:split_index], time_col.iloc[split_index:]

            # 정규화
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 모델 학습
            print("MLR")
            mlr = LinearRegression()
            mlr.fit(X_train_scaled, y_train)

            print("SVR")
            svr = SVR()
            svr.fit(X_train_scaled, y_train)

            print("LightGBM")
            lgb_model = lgb.LGBMRegressor()
            lgb_model.fit(X_train, y_train)

            print("MLP")
            mlp = MLPRegressor(random_state=42, max_iter=1000)
            mlp.fit(X_train_scaled, y_train)

            # 예측 결과를 t+1 위치에 저장
            df_result = X.copy()
            df_result.loc[:split_index - 1, 'pred_error_mlr'] = mlr.predict(X_train_scaled)
            df_result.loc[split_index:, 'pred_error_mlr'] = mlr.predict(X_test_scaled)

            df_result.loc[:split_index - 1, 'pred_error_svr'] = svr.predict(X_train_scaled)
            df_result.loc[split_index:, 'pred_error_svr'] = svr.predict(X_test_scaled)

            df_result.loc[:split_index - 1, 'pred_error_lgb'] = lgb_model.predict(X_train)
            df_result.loc[split_index:, 'pred_error_lgb'] = lgb_model.predict(X_test)

            df_result.loc[:split_index - 1, 'pred_error_mlp'] = mlp.predict(X_train_scaled)
            df_result.loc[split_index:, 'pred_error_mlp'] = mlp.predict(X_test_scaled)

            # shift
            for col in ['pred_error_mlr', 'pred_error_svr', 'pred_error_lgb', 'pred_error_mlp']:
                df_result[col] = df_result[col].shift(1)

            # 시간 복원
            X_test_result = df_result.copy()
            X_test_result.loc[:split_index - 1, 'Time'] = time_train.values
            X_test_result.loc[split_index:, 'Time'] = time_test.values
            X_test_result.dropna(subset=['pred_error_mlr', 'pred_error_svr', 'pred_error_lgb', 'pred_error_mlp'],
                                 inplace=True)

            reforecasted_df_list.append(X_test_result)

            # 모델 저장
            joblib.dump(mlr,
                        f"{path}/result_of_paper/ann/mlr_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            joblib.dump(svr,
                        f"{path}/result_of_paper/ann/svr_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            joblib.dump(mlp,
                        f"{path}/result_of_paper/ann/mlp_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            lgb_model.booster_.save_model(
                f"{path}/result_of_paper/ann/lgb_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.txt")

        # 재예측 값 계산 및 저장
        for forecasted_df in reforecasted_df_list:
            forecasted_df['reforecasted_PV_mlr'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_mlr']), 0)
            forecasted_df['reforecasted_PV_svr'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_svr']), 0)
            forecasted_df['reforecasted_PV_lgb'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_lgb']), 0)
            forecasted_df['reforecasted_PV_mlp'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_mlp']), 0)

            final_result_list.append(forecasted_df)

        for final_result_df, df_index in zip(final_result_list, site_names):
            final_result_df.to_csv(f"{path}/result_of_paper/ann/reforecast_result_{forecast_model}_{df_index}_.csv",
                                   index=False)

    return final_result_lists


# %% md
# %%
def print_out_reforecast_eval(final_result_lists, operation_hours_array):
    for forecast_model, final_result_list in zip(['mlr', 'svr', 'lgb', 'mlp'], final_result_lists):
        print(
            '================================================================================================================')
        print(f'forecast model : {forecast_model.upper()}')

        filtered_df_list = []
        for df, (start_str, end_str) in zip(final_result_list, operation_hours_array):
            split_index = int(len(df) * (1 - test_size))
            df_copy = df.iloc[split_index:].copy()
            df_copy['Time'] = pd.to_datetime(df_copy['Time'])

            start_time = time.fromisoformat(start_str)
            end_time = time.fromisoformat(end_str)

            df_copy = df_copy[df_copy['Time'].dt.time.between(start_time, end_time)]
            filtered_df_list.append(df_copy)

    for result_df, df_index in zip(filtered_df_list, site_names):
        print(
            '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(df_index)

        for reforecast_model in ['mlr', 'svr', 'lgb', 'mlp']:
            print(f'reforecast model : {reforecast_model.upper()}')

            mse = ((result_df['Power (MW)'] - result_df[f'reforecasted_PV_{reforecast_model}']) ** 2).mean()
            mae = (result_df['Power (MW)'] - result_df[f'reforecasted_PV_{reforecast_model}']).abs().mean()
            rmse = np.sqrt(mse)

            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")


final_result_lists = do_reforecast(targets, shifted_targets, features_list, result_for_reforecast_df_list)
print_out_reforecast_eval(final_result_lists, operation_hours_array)