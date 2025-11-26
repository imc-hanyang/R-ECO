"""
특징 선택 결과를 반영한 PV 발전량 예측(포캐스트) 파이프라인 모듈.

- 사이트별 / 모델별 최종 선택된 입력 피처를 정의
- 선택된 피처로 MLR, SVR, LGBM, MLP 예측 모델 학습 및 예측 수행
- 운전 시간대 기준으로 결과를 필터링하고 예측 성능(MAE, MSE, RMSE) 출력
"""

from datetime import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# -----------------------------------------------------------------------------
# 사이트별 MLR 기본 예측(forecast)에 사용되는 최종 선택 피처 목록
# -----------------------------------------------------------------------------
site1_features_mlr_forecast = ['Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8']
site2_features_mlr_forecast = [
    'Total solar irradiance (W/m2)', 'lag_4', 'lag_2', 'lag_5', 'hour_cos', 'lag_3', 'lag_6',
    'Power (MW)', 'same_time_max', 'lag_7', 'diff_75min', 'lag_1', 'diff_90min',
    'diff_15min', 'diff_45min', 'diff_60min', 'diff_105min', 'overall_mean', 'same_time_min',
    'hour_sin', 'same_time_std', 'Air temperature  (°C)'
]
site4_features_mlr_forecast = [
    'Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
    'same_weekday_std'
]
site5_features_mlr_forecast = [
    'lag_1', 'Power (MW)', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_105min', 'same_weekday_std', 'lag_8'
]
site6_features_mlr_forecast = [
    'Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5',
    'Total solar irradiance (W/m2)', 'lag_6', 'lag_7', 'lag_8', 'same_time_mean',
    'diff_90min'
]
site7_features_mlr_forecast = [
    'Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_6', 'lag_5', 'lag_7',
    'Total solar irradiance (W/m2)', 'same_time_mean', 'diff_105min', 'diff_75min',
    'hour_cos'
]
site8_features_mlr_forecast = [
    'lag_1', 'lag_2', 'Power (MW)', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_90min', 'diff_105min', 'same_time_max', 'hour_cos', 'diff_75min', 'lag_8',
    'diff_60min', 'same_weekday_mean', 'same_weekday_std', 'diff_45min'
]

# -----------------------------------------------------------------------------
# 사이트별 SVR 기본 예측(forecast)에 사용되는 최종 선택 피처 목록
# -----------------------------------------------------------------------------
site1_features_svr_forecast = [
    'Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5',
    'Total solar irradiance (W/m2)', 'lag_6', 'lag_7', 'diff_105min', 'diff_90min'
]
site2_features_svr_forecast = [
    'Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_105min', 'diff_90min'
]
site4_features_svr_forecast = [
    'Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_105min', 'diff_90min', 'diff_75min'
]
site5_features_svr_forecast = [
    'Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_105min', 'Total solar irradiance (W/m2)', 'diff_90min'
]
site6_features_svr_forecast = [
    'Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_105min', 'same_time_max'
]
site7_features_svr_forecast = [
    'Power (MW)', 'lag_2', 'lag_1', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_105min', 'diff_75min'
]
site8_features_svr_forecast = [
    'Power (MW)', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'diff_105min', 'diff_90min', 'diff_75min', 'same_time_mean'
]

# -----------------------------------------------------------------------------
# 사이트별 LightGBM 기본 예측(forecast)에 사용되는 최종 선택 피처 목록
# -----------------------------------------------------------------------------
site1_features_lgb_forecast = ['Power (MW)', 'Total solar irradiance (W/m2)']
site2_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site4_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site5_features_lgb_forecast = ['Power (MW)', 'lag_1', 'hour_sin', 'lag_2']
site6_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site7_features_lgb_forecast = ['Power (MW)', 'hour_sin']
site8_features_lgb_forecast = ['Power (MW)', 'lag_1', 'hour_sin']

# -----------------------------------------------------------------------------
# 사이트별 MLP 기본 예측(forecast)에 사용되는 최종 선택 피처 목록
# -----------------------------------------------------------------------------
site1_features_mlp_forecast = [
    'Total solar irradiance (W/m2)', 'lag_4', 'lag_2', 'lag_5', 'hour_cos', 'lag_3', 'lag_6',
    'Power (MW)', 'same_time_max', 'lag_7', 'diff_75min', 'lag_1', 'diff_90min',
    'diff_15min', 'diff_45min'
]
site2_features_mlp_forecast = [
    'hour_cos', 'lag_4', 'lag_6', 'lag_3', 'lag_2', 'Power (MW)', 'lag_1', 'lag_5', 'lag_7',
    'diff_75min', 'diff_105min', 'same_time_mean', 'diff_90min'
]
site4_features_mlp_forecast = [
    'lag_5', 'lag_2', 'lag_3', 'lag_4', 'lag_1', 'Power (MW)', 'lag_6', 'hour_cos', 'lag_7',
    'hour_sin', 'diff_90min'
]
site5_features_mlp_forecast = [
    'lag_3', 'lag_2', 'lag_4', 'lag_5', 'hour_cos', 'lag_1', 'Power (MW)', 'lag_6',
    'hour_sin', 'lag_7', 'diff_75min', 'diff_105min'
]
site6_features_mlp_forecast = [
    'Power (MW)', 'hour_sin', 'lag_2', 'hour_cos', 'diff_15min',
    'Total solar irradiance (W/m2)', 'diff_105min'
]
site7_features_mlp_forecast = [
    'hour_cos', 'lag_2', 'lag_3', 'lag_6', 'Power (MW)', 'lag_5', 'lag_4', 'lag_7',
    'same_time_mean', 'diff_105min', 'lag_1', 'diff_75min', 'diff_45min', 'diff_90min',
    'hour_sin'
]
site8_features_mlp_forecast = [
    'Power (MW)', 'lag_2', 'lag_3', 'hour_cos', 'lag_5', 'lag_7', 'lag_6', 'lag_1', 'lag_4',
    'hour_sin', 'diff_105min', 'diff_75min', 'same_time_std', 'diff_90min', 'diff_60min',
    'diff_45min'
]

# -----------------------------------------------------------------------------
# 사이트별로 [MLR, SVR, LGB, MLP] 피처 리스트를 묶은 구조
# -----------------------------------------------------------------------------
site1_features_list = [
    site1_features_mlr_forecast,
    site1_features_svr_forecast,
    site1_features_lgb_forecast,
    site1_features_mlp_forecast,
]
site2_features_list = [
    site2_features_mlr_forecast,
    site2_features_svr_forecast,
    site2_features_lgb_forecast,
    site2_features_mlp_forecast,
]
site4_features_list = [
    site4_features_mlr_forecast,
    site4_features_svr_forecast,
    site4_features_lgb_forecast,
    site4_features_mlp_forecast,
]
site5_features_list = [
    site5_features_mlr_forecast,
    site5_features_svr_forecast,
    site5_features_lgb_forecast,
    site5_features_mlp_forecast,
]
site6_features_list = [
    site6_features_mlr_forecast,
    site6_features_svr_forecast,
    site6_features_lgb_forecast,
    site6_features_mlp_forecast,
]
site7_features_list = [
    site7_features_mlr_forecast,
    site7_features_svr_forecast,
    site7_features_lgb_forecast,
    site7_features_mlp_forecast,
]
site8_features_list = [
    site8_features_mlr_forecast,
    site8_features_svr_forecast,
    site8_features_lgb_forecast,
    site8_features_mlp_forecast,
]

# 전체 사이트에 대한 피처 리스트(사이트별 → 모델별)
features_list = [
    site1_features_list,
    site2_features_list,
    site4_features_list,
    site5_features_list,
    site6_features_list,
    site7_features_list,
    site8_features_list,
]


def do_forecast(
    df_list: list,
    site_names: list,
    features_list: list,
    target: str,
    shifted_target: str,
    forecast_models: list,
    test_size: float,
    path: str,
):
    """
    선택된 피처를 사용해 사이트별로 MLR/SVR/LGB/MLP 예측 모델을 학습하고 예측을 수행하는 함수.

    Args:
        df_list (list): 사이트별 입력 DataFrame 리스트.
        site_names (list): 사이트 ID 리스트.
        features_list (list): 사이트별 모델별 피처 리스트 (sites × 4).
        target (str): 원래 타깃 컬럼명 (예: 'Power (MW)').
        shifted_target (str): t+1 예측을 위한 시프트된 타깃 컬럼명.
        forecast_models (list): 예측 모델 이름 리스트 (['mlr', 'svr', 'lgb', 'mlp']).
        test_size (float): 테스트 데이터 비율 (0~1 사이).
        path (str): 결과 저장 기본 경로.

    Returns:
        list: 재예측에 사용하기 위한 예측 결과 DataFrame 리스트.
    """
    forecasted_df_list = []

    for df, df_index, features in zip(df_list, site_names, features_list):
        print('----------------------------------------------------------------------------------------------')
        print(df_index)

        # 타깃을 한 스텝 뒤로 시프트하여 예측 대상(shifted_target) 생성
        df[shifted_target] = df[target].shift(-1)
        df = df.dropna().reset_index(drop=True)

        # 전체 피처/타깃/시간 컬럼 복사
        X_total = df.copy()
        y = df[shifted_target]
        time_col = df['Time']

        # 시계열 순서를 유지한 채 train/test 분할
        split_index = int(len(X_total) * (1 - test_size))
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        time_train, time_test = time_col.iloc[:split_index], time_col.iloc[split_index:]

        # 스케일러 초기화 (모델별로 재사용)
        scaler = StandardScaler()

        # -------------------- MLR --------------------
        print("MLR")
        filtered_features = [col for col in features[0] if col in df.columns]
        X = df[filtered_features]
        X_train_mlr, X_test_mlr = X.iloc[:split_index], X.iloc[split_index:]
        X_train_scaled_mlr = scaler.fit_transform(X_train_mlr)
        X_test_scaled_mlr = scaler.transform(X_test_mlr)

        mlr = LinearRegression()
        mlr.fit(X_train_scaled_mlr, y_train)

        # -------------------- SVR --------------------
        print("SVR")
        filtered_features = [col for col in features[1] if col in df.columns]
        X = df[filtered_features]
        X_train_svr, X_test_svr = X.iloc[:split_index], X.iloc[split_index:]
        X_train_scaled_svr = scaler.fit_transform(X_train_svr)
        X_test_scaled_svr = scaler.transform(X_test_svr)

        svr = SVR()
        svr.fit(X_train_scaled_svr, y_train)

        # -------------------- LightGBM --------------------
        print("LightGBM")
        filtered_features = [col for col in features[2] if col in df.columns]
        X = df[filtered_features]
        X_train_lgb, X_test_lgb = X.iloc[:split_index], X.iloc[split_index:]

        lgb_model = lgb.LGBMRegressor()
        lgb_model.fit(X_train_lgb, y_train)

        # -------------------- MLP --------------------
        print("MLP")
        filtered_features = [col for col in features[3] if col in df.columns]
        X = df[filtered_features]
        X_train_mlp, X_test_mlp = X.iloc[:split_index], X.iloc[split_index:]
        X_train_scaled_mlp = scaler.fit_transform(X_train_mlp)
        X_test_scaled_mlp = scaler.transform(X_test_mlp)

        mlp = MLPRegressor(random_state=42, max_iter=1000)
        mlp.fit(X_train_scaled_mlp, y_train)

        # -------------------- 예측 결과를 t+1 위치에 저장 --------------------
        df_result = X_total.copy()
        df_result.loc[:split_index - 1, 'pred_mlr'] = mlr.predict(X_train_scaled_mlr)
        df_result.loc[split_index:, 'pred_mlr'] = mlr.predict(X_test_scaled_mlr)

        df_result.loc[:split_index - 1, 'pred_svr'] = svr.predict(X_train_scaled_svr)
        df_result.loc[split_index:, 'pred_svr'] = svr.predict(X_test_scaled_svr)

        df_result.loc[:split_index - 1, 'pred_lgb'] = lgb_model.predict(X_train_lgb)
        df_result.loc[split_index:, 'pred_lgb'] = lgb_model.predict(X_test_lgb)

        df_result.loc[:split_index - 1, 'pred_mlp'] = mlp.predict(X_train_scaled_mlp)
        df_result.loc[split_index:, 'pred_mlp'] = mlp.predict(X_test_scaled_mlp)

        # t 시점에 예측값이 오도록 1 스텝 뒤로 시프트
        df_result['pred_mlr'] = df_result['pred_mlr'].shift(1)
        df_result['pred_svr'] = df_result['pred_svr'].shift(1)
        df_result['pred_lgb'] = df_result['pred_lgb'].shift(1)
        df_result['pred_mlp'] = df_result['pred_mlp'].shift(1)

        # 시간 정보 복원 (train/test 구간 기준)
        X_test_result = df_result.copy()
        X_test_result.loc[:split_index - 1, 'Time'] = time_train.values
        X_test_result.loc[split_index:, 'Time'] = time_test.values

        # 예측값 모두 존재하는 행만 사용
        X_test_result.dropna(subset=['pred_mlr', 'pred_svr', 'pred_lgb', 'pred_mlp'], inplace=True)

        # 사이트별 예측 결과 리스트에 추가
        forecasted_df_list.append(X_test_result)

        # 학습된 모델 저장 (피처 선택 기반)
        joblib.dump(
            mlr,
            f"{path}/result_of_paper/feature_selection/forecast/mlr_forecasting_model_{str(df_index)}.joblib",
        )
        joblib.dump(
            svr,
            f"{path}/result_of_paper/feature_selection//forecast/svr_forecasting_model_{str(df_index)}.joblib",
        )
        lgb_model.booster_.save_model(
            f"{path}/result_of_paper/feature_selection/forecast/lgb_forecasting_model_{str(df_index)}.txt",
        )
        joblib.dump(
            mlp,
            f"{path}/result_of_paper/feature_selection/forecast/mlp_forecasting_model_{str(df_index)}.joblib",
        )

    # -------------------- 예측값 후처리 및 에러 계산 --------------------
    result_for_reforecast_df_list = []
    for site_num, forecasted_df in zip(site_names, forecasted_df_list):
        for model in forecast_models:
            # 음수 예측값을 0으로 보정
            forecasted_df[f'pred_{model}'] = forecasted_df[f'pred_{model}'].apply(lambda x: max(x, 0))

            # 예측 오차(error_model) = pred_model - 실제값(target)
            forecasted_df[f'error_{model}'] = forecasted_df.apply(
                lambda row: (row[f'pred_{model}'] - row[target]),
                axis=1,
            )

        result_for_reforecast_df_list.append(forecasted_df)

    return result_for_reforecast_df_list


def save_forecast_result(result_for_reforecast_df_list: list, site_names: list, path: str):
    """
    피처 선택 후 예측 결과 DataFrame을 CSV 파일로 저장하는 함수.

    Args:
        result_for_reforecast_df_list (list): 사이트별 예측/오차 결과 DataFrame 리스트.
        site_names (list): 사이트 ID 리스트.
        path (str): 저장 기본 경로.
    """
    for final_result_df, df_index in zip(result_for_reforecast_df_list, site_names):
        final_result_df.to_csv(
            f"{path}/result_of_paper/feature_selection/forecast/forecast_result_{df_index}_.csv",
            index=False,
        )


def filter_forecast_result_by_operation_times(
    result_for_reforecast_df_list,
    operation_hours_array,
    test_size: float,
):
    """
    운전 시간대(Operation hours)에 해당하는 시간 구간만 필터링하는 함수.

    Args:
        result_for_reforecast_df_list (list): 사이트별 예측/오차 결과 DataFrame 리스트.
        operation_hours_array (list): 사이트별 운전 시간대 튜플 리스트 (start_str, end_str).
        test_size (float): 테스트 데이터 비율 (0~1 사이).

    Returns:
        list: 운전 시간대 기준으로 필터링된 DataFrame 리스트.
    """
    filtered_df_list = []

    for df, (start_str, end_str) in zip(result_for_reforecast_df_list, operation_hours_array):
        # 테스트 구간만 사용 (train/test 시점 기준 split)
        split_index = int(len(df) * (1 - test_size))
        df_copy = df.iloc[split_index:].copy()
        df_copy['Time'] = pd.to_datetime(df_copy['Time'])

        start_time = time.fromisoformat(start_str)
        end_time = time.fromisoformat(end_str)

        # Time 컬럼의 시각 정보만 사용해 운전 시간대 필터링
        df_copy = df_copy[df_copy['Time'].dt.time.between(start_time, end_time)]
        filtered_df_list.append(df_copy)

    return filtered_df_list


def print_out_forecast_eval(filtered_df_list, site_names, forecast_models: list):
    """
    운전 시간대 기준으로 필터링된 데이터에 대해 Forecast 성능(MAE/MSE/RMSE)을 출력하는 함수.

    Args:
        filtered_df_list (list): 운전 시간대로 필터링된 DataFrame 리스트.
        site_names (list): 사이트 ID 리스트.
        forecast_models (list): 예측 모델 이름 리스트 (['mlr', 'svr', 'lgb', 'mlp']).
    """
    for result_df, df_index in zip(filtered_df_list, site_names):
        print(
            '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        )
        print(df_index)

        for forecast_model in forecast_models:
            print(f'reforecast model : {forecast_model.upper()}')

            mse = (result_df[f'error_{forecast_model}'] ** 2).mean()
            mae = result_df[f'error_{forecast_model}'].abs().mean()
            rmse = np.sqrt(mse)

            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")


def do_forecast_after_feature_selection(
    site_names: list,
    path: str,
    operation_hours_array: list,
    test_size: float,
    target: str,
    shifted_target: str,
    forecast_models: list,
):
    """
    (1) lag/feature가 추가된 CSV를 로드하고,
    (2) 선택된 피처 기반 Forecast를 수행하며,
    (3) 운전 시간대 기준으로 필터링 후 성능을 출력하는 최종 실행 함수.

    Args:
        site_names (list): 사이트 ID 리스트.
        path (str): 데이터 및 결과가 저장된 기본 경로.
        operation_hours_array (list): 사이트별 운전 시간대 튜플 리스트.
        test_size (float): 테스트 데이터 비율.
        target (str): 원래 타깃 컬럼명 (예: 'Power (MW)').
        shifted_target (str): t+1 예측을 위한 시프트된 타깃 이름.
        forecast_models (list): 예측 모델 이름 리스트 (['mlr', 'svr', 'lgb', 'mlp']).
    """
    df_list = []
    for df_index in site_names:
        # 각 사이트별 lag/feature가 추가된 CSV를 로드
        df_list.append(pd.read_csv(f'{path}/lag_added_dataset/{str(df_index)}.csv'))

    # 선택된 피처를 활용하여 Forecast 수행
    result_for_reforecast_df_list = do_forecast(
        df_list,
        site_names,
        features_list,
        target,
        shifted_target,
        forecast_models,
        test_size,
        path,
    )

    # 예측 결과 CSV 저장
    save_forecast_result(result_for_reforecast_df_list, site_names, path)

    # 운전 시간대별로 필터링 후 성능 출력
    filtered_df_list = filter_forecast_result_by_operation_times(
        result_for_reforecast_df_list,
        operation_hours_array,
        test_size,
    )
    print_out_forecast_eval(filtered_df_list, site_names, forecast_models)
