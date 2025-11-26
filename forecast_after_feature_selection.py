from datetime import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# mlr forcecast reforecast features
site1_features_mlr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8']
site2_features_mlr_forecast = ['Total solar irradiance (W/m2)', 'lag_4', 'lag_2', 'lag_5', 'hour_cos', 'lag_3', 'lag_6',
                               'Power (MW)', 'same_time_max', 'lag_7', 'diff_75min', 'lag_1', 'diff_90min',
                               'diff_15min', 'diff_45min', 'diff_60min', 'diff_105min', 'overall_mean', 'same_time_min',
                               'hour_sin', 'same_time_std', 'Air temperature  (°C)']
site4_features_mlr_forecast = ['Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
                               'same_weekday_std']
site5_features_mlr_forecast = ['lag_1', 'Power (MW)', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_105min', 'same_weekday_std', 'lag_8']
site6_features_mlr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5',
                               'Total solar irradiance (W/m2)', 'lag_6', 'lag_7', 'lag_8', 'same_time_mean',
                               'diff_90min']
site7_features_mlr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_6', 'lag_5', 'lag_7',
                               'Total solar irradiance (W/m2)', 'same_time_mean', 'diff_105min', 'diff_75min',
                               'hour_cos']
site8_features_mlr_forecast = ['lag_1', 'lag_2', 'Power (MW)', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_90min', 'diff_105min', 'same_time_max', 'hour_cos', 'diff_75min', 'lag_8',
                               'diff_60min', 'same_weekday_mean', 'same_weekday_std', 'diff_45min']

site1_features_svr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5',
                               'Total solar irradiance (W/m2)', 'lag_6', 'lag_7', 'diff_105min', 'diff_90min']
site2_features_svr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_105min', 'diff_90min']
site4_features_svr_forecast = ['Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_105min', 'diff_90min', 'diff_75min']
site5_features_svr_forecast = ['Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_105min', 'Total solar irradiance (W/m2)', 'diff_90min']
site6_features_svr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_105min', 'same_time_max']
site7_features_svr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_105min', 'diff_75min']
site8_features_svr_forecast = ['Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                               'diff_105min', 'diff_90min', 'diff_75min', 'same_time_mean']

site1_features_lgb_forecast = ['Power (MW)', 'Total solar irradiance (W/m2)']
site2_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site4_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site5_features_lgb_forecast = ['Power (MW)', 'lag_1', 'hour_sin', 'lag_2']
site6_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site7_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site8_features_lgb_forecast = ['Power (MW)', 'lag_1', 'hour_sin']

site1_features_mlp_forecast = ['Total solar irradiance (W/m2)', 'lag_4', 'lag_2', 'lag_5', 'hour_cos', 'lag_3', 'lag_6',
                               'Power (MW)', 'same_time_max', 'lag_7', 'diff_75min', 'lag_1', 'diff_90min',
                               'diff_15min', 'diff_45min']
site2_features_mlp_forecast = ['hour_cos', 'lag_4', 'lag_6', 'lag_3', 'lag_2', 'Power (MW)', 'lag_1', 'lag_5', 'lag_7',
                               'diff_75min', 'diff_105min', 'same_time_mean', 'diff_90min']
site4_features_mlp_forecast = ['lag_5', 'lag_2', 'lag_3', 'lag_4', 'lag_1', 'Power (MW)', 'lag_6', 'hour_cos', 'lag_7',
                               'hour_sin', 'diff_90min']
site5_features_mlp_forecast = ['lag_3', 'lag_2', 'lag_4', 'lag_5', 'hour_cos', 'lag_1', 'Power (MW)', 'lag_6',
                               'hour_sin', 'lag_7', 'diff_75min', 'diff_105min']
site6_features_mlp_forecast = ['Power (MW)', 'hour_sin', 'lag_2', 'hour_cos', 'diff_15min',
                               'Total solar irradiance (W/m2)', 'diff_105min']
site7_features_mlp_forecast = ['hour_cos', 'lag_2', 'lag_3', 'lag_6', 'Power (MW)', 'lag_5', 'lag_4', 'lag_7',
                               'same_time_mean', 'diff_105min', 'lag_1', 'diff_75min', 'diff_45min', 'diff_90min',
                               'hour_sin']
site8_features_mlp_forecast = ['Power (MW)', 'lag_2', 'lag_3', 'hour_cos', 'lag_5', 'lag_7', 'lag_6', 'lag_1', 'lag_4',
                               'hour_sin', 'diff_105min', 'diff_75min', 'same_time_std', 'diff_90min', 'diff_60min',
                               'diff_45min']

site1_features_list = [site1_features_mlr_forecast, site1_features_svr_forecast, site1_features_lgb_forecast,
                       site1_features_mlp_forecast]
site2_features_list = [site2_features_mlr_forecast, site2_features_svr_forecast, site2_features_lgb_forecast,
                       site2_features_mlp_forecast]
site4_features_list = [site4_features_mlr_forecast, site4_features_svr_forecast, site4_features_lgb_forecast,
                       site4_features_mlp_forecast]
site5_features_list = [site5_features_mlr_forecast, site5_features_svr_forecast, site5_features_lgb_forecast,
                       site5_features_mlp_forecast]
site6_features_list = [site6_features_mlr_forecast, site6_features_svr_forecast, site6_features_lgb_forecast,
                       site6_features_mlp_forecast]
site7_features_list = [site7_features_mlr_forecast, site7_features_svr_forecast, site7_features_lgb_forecast,
                       site7_features_mlp_forecast]
site8_features_list = [site8_features_mlr_forecast, site8_features_svr_forecast, site8_features_lgb_forecast,
                       site8_features_mlp_forecast]

features_list = [site1_features_list, site2_features_list, site4_features_list, site5_features_list,
                 site6_features_list, site7_features_list, site8_features_list]


def do_forecast(df_list: list, site_names: list, features_list: list, target: str, shifted_target: str,
                forecast_models:list, test_size: float, path: str):
    forecasted_df_list = []
    for df, df_index, features in zip(df_list, site_names, features_list):
        print('----------------------------------------------------------------------------------------------')
        print(df_index)

        df[shifted_target] = df[target].shift(-1)
        df = df.dropna().reset_index(drop=True)

        X_total = df.copy()
        y = df[shifted_target]
        time_col = df['Time']

        # 시계열 순서 기반 분할
        split_index = int(len(X_total) * (1 - test_size))

        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        time_train, time_test = time_col.iloc[:split_index], time_col.iloc[split_index:]

        # 정규화
        scaler = StandardScaler()

        # MLR
        print("MLR")

        filtered_features = [col for col in features[0] if col in df.columns]
        X = df[filtered_features]
        X_train_mlr, X_test_mlr = X.iloc[:split_index], X.iloc[split_index:]
        X_train_scaled_mlr = scaler.fit_transform(X_train_mlr)
        X_test_scaled_mlr = scaler.transform(X_test_mlr)

        mlr = LinearRegression()
        mlr.fit(X_train_scaled_mlr, y_train)

        # SVR
        print("SVR")

        filtered_features = [col for col in features[1] if col in df.columns]
        X = df[filtered_features]
        X_train_svr, X_test_svr = X.iloc[:split_index], X.iloc[split_index:]
        X_train_scaled_svr = scaler.fit_transform(X_train_svr)
        X_test_scaled_svr = scaler.transform(X_test_svr)

        svr = SVR()
        svr.fit(X_train_scaled_svr, y_train)

        # LightGBM
        print("LightGBM")

        filtered_features = [col for col in features[2] if col in df.columns]
        X = df[filtered_features]
        X_train_lgb, X_test_lgb = X.iloc[:split_index], X.iloc[split_index:]

        lgb_model = lgb.LGBMRegressor()
        lgb_model.fit(X_train_lgb, y_train)

        # MLP
        print("MLP")

        filtered_features = [col for col in features[3] if col in df.columns]
        X = df[filtered_features]
        X_train_mlp, X_test_mlp = X.iloc[:split_index], X.iloc[split_index:]
        X_train_scaled_mlp = scaler.fit_transform(X_train_mlp)
        X_test_scaled_mlp = scaler.transform(X_test_mlp)

        mlp = MLPRegressor(random_state=42, max_iter=1000)
        mlp.fit(X_train_scaled_mlp, y_train)

        # 예측 결과를 t+1 위치에 저장
        df_result = X_total.copy()
        df_result.loc[:split_index - 1, 'pred_mlr'] = mlr.predict(X_train_scaled_mlr)
        df_result.loc[split_index:, 'pred_mlr'] = mlr.predict(X_test_scaled_mlr)
        df_result.loc[:split_index - 1, 'pred_svr'] = svr.predict(X_train_scaled_svr)
        df_result.loc[split_index:, 'pred_svr'] = svr.predict(X_test_scaled_svr)
        df_result.loc[:split_index - 1, 'pred_lgb'] = lgb_model.predict(X_train_lgb)
        df_result.loc[split_index:, 'pred_lgb'] = lgb_model.predict(X_test_lgb)
        df_result.loc[:split_index - 1, 'pred_mlp'] = mlp.predict(X_train_scaled_mlp)
        df_result.loc[split_index:, 'pred_mlp'] = mlp.predict(X_test_scaled_mlp)

        df_result['pred_mlr'] = df_result['pred_mlr'].shift(1)
        df_result['pred_svr'] = df_result['pred_svr'].shift(1)
        df_result['pred_lgb'] = df_result['pred_lgb'].shift(1)
        df_result['pred_mlp'] = df_result['pred_mlp'].shift(1)

        # 결과 추출
        X_test_result = df_result.copy()
        X_test_result.loc[:split_index - 1, 'Time'] = time_train.values
        X_test_result.loc[split_index:, 'Time'] = time_test.values
        X_test_result.dropna(subset=['pred_mlr', 'pred_svr', 'pred_lgb', 'pred_mlp'], inplace=True)

        forecasted_df_list.append(X_test_result)

        joblib.dump(mlr, f"{path}/result_of_paper/feature_selection/forecast/mlr_forecasting_model_{str(df_index)}.joblib")
        joblib.dump(svr, f"{path}/result_of_paper/feature_selection//forecast/svr_forecasting_model_{str(df_index)}.joblib")
        lgb_model.booster_.save_model(
            f"{path}/result_of_paper/feature_selection/forecast/lgb_forecasting_model_{str(df_index)}.txt")
        joblib.dump(mlp, f"{path}/result_of_paper/feature_selection/forecast/mlp_forecasting_model_{str(df_index)}.joblib")
    result_for_reforecast_df_list = []
    for site_num, forecasted_df in zip(site_names, forecasted_df_list):
        for model in forecast_models:
            forecasted_df[f'pred_{model}'] = forecasted_df[f'pred_{model}'].apply(lambda x: max(x, 0))
            forecasted_df[f'error_{model}'] = forecasted_df.apply(
                lambda row: (row[f'pred_{model}'] - row[target]),
                axis=1
            )

        result_for_reforecast_df_list.append(forecasted_df)

    return result_for_reforecast_df_list


def save_forecast_result(result_for_reforecast_df_list: list, site_names: list, path: str):
    for final_result_df, df_index in zip(result_for_reforecast_df_list, site_names):
        final_result_df.to_csv(f"{path}/result_of_paper/feature_selection/forecast/forecast_result_{df_index}_.csv",
                               index=False)


def filter_forecast_result_by_operation_times(result_for_reforecast_df_list, operation_hours_array, test_size: float):
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


def print_out_forecast_eval(filtered_df_list, site_names, forecast_models: list):
    for result_df, df_index in zip(filtered_df_list, site_names):
        print(
            '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(df_index)

        for forecast_model in forecast_models:
            print(f'reforecast model : {forecast_model.upper()}')

            mse = (result_df[f'error_{forecast_model}'] ** 2).mean()
            mae = result_df[f'error_{forecast_model}'].abs().mean()
            rmse = np.sqrt(mse)

            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")


def do_forecast_after_feature_selection(site_names: list, path: str, operation_hours_array: list, test_size: float,
                                        target: str, shifted_target: str, forecast_models: list):
    df_list = []
    for df_index in site_names:
        df_list.append(pd.read_csv(f'{path}/lag_added_dataset/f"{str(df_index)}.csv'))
    result_for_reforecast_df_list = do_forecast(df_list, site_names, features_list, target, shifted_target,forecast_models,
                                                test_size, path)
    save_forecast_result(result_for_reforecast_df_list, site_names, path)
    filtered_df_list = filter_forecast_result_by_operation_times(result_for_reforecast_df_list, operation_hours_array,
                                                                 test_size)
    print_out_forecast_eval(filtered_df_list, site_names, forecast_models)
