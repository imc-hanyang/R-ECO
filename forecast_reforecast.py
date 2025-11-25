"""
PV 발전량 예측 및 재예측(Reforecast)을 수행하는 모듈.

- 전처리된 데이터(lag/feature 추가된 CSV)를 읽어 DataFrame 리스트로 생성
- MLR, SVR, LightGBM, MLP 기반의 기본 예측(Base Forecast) 수행
- 예측 오차에 대해 재예측(Reforecast)을 수행하여 보정된 PV 발전량 산출
- 운전 시간대(Operation hours)에 따른 필터링 및 성능 평가 지표(MAE, MSE, RMSE) 출력
"""

import os
import pandas as pd
import joblib
import numpy as np
from datetime import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb

from feature_lists import forecast_mlr_features, forecast_svr_features, forecast_lgb_features, \
    forecast_mlp_features


def get_feature_added_dataframe(path: str) -> list:
    """
    lag 및 다양한 피처가 이미 추가된 CSV 파일들을 읽어
    DataFrame 리스트로 반환하는 함수.

    Args:
        path (str): 전처리된 데이터가 저장된 상위 디렉터리 경로.
                    내부에서 path + '/lag_added_dataset'를 사용하여 CSV를 탐색함.

    Returns:
        list: 각 CSV 파일을 읽어 만든 pandas DataFrame 객체들의 리스트.
    """
    files = [os.path.join(path + '/lag_added_dataset', file) for file in os.listdir(path + '/lag_added_dataset') if
             file.endswith('.csv')]
    origin_df_list = []
    files.sort()
    for file in files:
        print(file)
        df = pd.read_csv(file)

        # 사용하지 않을 컬럼들을 순차적으로 제거
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

    return origin_df_list


def do_base_forecast(origin_df_list: list, site_names: list, features: list, target: str, shifted_column: str,
                     test_size: float, path: str):
    """
    MLR, SVR, LightGBM, MLP 네 가지 모델로 기본 예측(Base Forecast)을 수행하는 함수.

    처리 내용:
        1. 다음 시점 예측을 위해 타깃 컬럼을 한 스텝 뒤로 shift하여 shifted_column에 저장.
        2. 시계열 순서를 유지한 채 train/test를 분할(test_size 비율).
        3. 정규화(StandardScaler) 후 MLR, SVR, MLP 모델 학습 및 예측.
        4. LightGBM 모델은 비스케일 입력으로 별도로 학습 및 예측.
        5. 예측 결과를 t+1 시점에 대응시키기 위해 예측값을 한 스텝 shift.
        6. 예측 결과 및 오차(error)를 포함한 DataFrame 리스트 반환.
        7. 학습된 모델들을 joblib / 텍스트 파일로 저장.

    Args:
        origin_df_list (list): 각 발전소(site)에 대한 원본(전처리 완료) DataFrame 리스트.
        site_names (list): 각 DataFrame에 대응되는 사이트 이름 리스트.
        features (list): 예측에 사용할 피처 컬럼 이름 리스트.
        target (str): 예측 대상이 되는 타깃 컬럼명 (예: 'Power (MW)').
        shifted_column (str): 다음 시점 타깃을 저장할 컬럼명.
        test_size (float): 테스트 데이터 비율 (0~1 사이).
        path (str): 모델 및 결과를 저장할 상위 디렉터리 경로.

    Returns:
        list: 예측값(pred_*)과 오차(error_*)가 포함된 DataFrame 리스트.
    """
    forecasted_df_list = []

    for i in range(len(origin_df_list)):
        print("===============================================================")
        print(site_names[i])
        df = origin_df_list[i].copy()

        # 다음 시점 예측을 위한 타겟 시프트
        df[shifted_column] = df[target].shift(-1)
        df = df.dropna().reset_index(drop=True)

        # 실제 DataFrame에 존재하는 피처만 사용 (필터링)
        filtered_features = [col for col in features if col in df.columns]
        X = df[filtered_features]
        y = df[shifted_column]
        time_col = df['Time(year-month-day h:m:s)']

        # 시계열 순서 기반 분할 인덱스 계산
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        time_train, time_test = time_col.iloc[:split_index], time_col.iloc[split_index:]

        # 정규화 (스케일링)
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

        # 예측 결과 저장용 DataFrame 생성
        df_result = X.copy()
        df_result.loc[:split_index - 1, 'pred_mlr'] = mlr.predict(X_train_scaled)
        df_result.loc[split_index:, 'pred_mlr'] = mlr.predict(X_test_scaled)

        df_result.loc[:split_index - 1, 'pred_svr'] = svr.predict(X_train_scaled)
        df_result.loc[split_index:, 'pred_svr'] = svr.predict(X_test_scaled)

        df_result.loc[:split_index - 1, 'pred_lgb'] = lgb_model.predict(X_train)
        df_result.loc[split_index:, 'pred_lgb'] = lgb_model.predict(X_test)

        df_result.loc[:split_index - 1, 'pred_mlp'] = mlp.predict(X_train_scaled)
        df_result.loc[split_index:, 'pred_mlp'] = mlp.predict(X_test_scaled)

        # 예측값을 t+1 시점에 위치시키기 위해 한 스텝 앞으로 shift
        for col in ['pred_mlr', 'pred_svr', 'pred_lgb', 'pred_mlp']:
            df_result[col] = df_result[col].shift(1)

        # 시간 컬럼 복원
        X_test_result = df_result.copy()
        X_test_result.loc[:split_index - 1, 'Time'] = time_train.values
        X_test_result.loc[split_index:, 'Time'] = time_test.values
        X_test_result.dropna(subset=['pred_mlr', 'pred_svr', 'pred_lgb', 'pred_mlp'], inplace=True)

        forecasted_df_list.append(X_test_result)

        # 모델 저장 (base forecast 모델)
        joblib.dump(mlr, f"{path}/result_of_paper/forecast/mlr_forecasting_model_{str(site_names[i])}.joblib")
        joblib.dump(svr, f"{path}/result_of_paper/forecast/svr_forecasting_model_{str(site_names[i])}.joblib")
        joblib.dump(mlp, f"{path}/result_of_paper/forecast/mlp_forecasting_model_{str(site_names[i])}.joblib")
        lgb_model.booster_.save_model(f"{path}/result_of_paper/forecast/lgb_forecasting_model_{str(site_names[i])}.txt")

    # 오차(error) 계산 및 결과 리스트 구성
    result_for_reforecast_df_list = []
    for forecasted_df in forecasted_df_list:
        for model in ['mlr', 'svr', 'lgb', 'mlp']:
            # 음수 예측값을 0 이상으로 클리핑
            forecasted_df[f'pred_{model}'] = forecasted_df[f'pred_{model}'].apply(lambda x: max(x, 0))
            # 예측 오차 = 예측값 - 실제값
            forecasted_df[f'error_{model}'] = forecasted_df.apply(
                lambda row: (row[f'pred_{model}'] - row['Power (MW)']),
                axis=1
            )

        result_for_reforecast_df_list.append(forecasted_df)

    return result_for_reforecast_df_list


# set operation hours for each selected df
def save_forecast_result(result_for_reforecast_df_list: list, site_names: list, path) -> None:
    """
    기본 예측(Base Forecast) 결과를 CSV 파일로 저장하는 함수.

    Args:
        result_for_reforecast_df_list (list): 예측 및 오차 정보가 포함된 DataFrame 리스트.
        site_names (list): 각 DataFrame에 대응되는 사이트 이름 리스트.
        path: 결과를 저장할 상위 디렉터리 경로.

    Returns:
        None
    """
    for final_result_df, df_index in zip(result_for_reforecast_df_list, site_names):
        final_result_df.to_csv(f"{path}/result_of_paper/forecast/forecast_result/{str(df_index)}.csv",
                               index=False)


def filter_by_operation_hours(result_for_reforecast_df_list: list, operation_hours_array: list,
                              test_size: float) -> list:
    """
    운전 시간대(Operation hours)에 해당하는 시점만 필터링하는 함수.

    Args:
        result_for_reforecast_df_list (list): 예측/오차 정보가 포함된 DataFrame 리스트.
        operation_hours_array (list): 각 사이트별 (시작시각 문자열, 종료시각 문자열) 튜플 리스트.
                                      예: [('06:00', '18:00'), ...]
        test_size (float): base forecast 단계에서 사용한 테스트 비율(시계열 분할 기준 유지용).

    Returns:
        list: 운전 시간대에 해당하는 행만 남긴 DataFrame 리스트.
    """
    filtered_df_list = []

    for df, (start_str, end_str) in zip(result_for_reforecast_df_list, operation_hours_array):
        # base forecast와 동일한 split_index 기준으로 test 구간만 사용
        split_index = int(len(df) * (1 - test_size))
        df_copy = df.iloc[split_index:].copy()
        df_copy['Time'] = pd.to_datetime(df_copy['Time'])

        # 문자열을 time 객체로 변환
        start_time = time.fromisoformat(start_str)
        end_time = time.fromisoformat(end_str)

        # 시각이 운전 시간대에 포함되는 행만 필터링
        df_copy = df_copy[df_copy['Time'].dt.time.between(start_time, end_time)]
        filtered_df_list.append(df_copy)

    return filtered_df_list


def print_out_forecast_eval_result(filtered_df_list, site_names):
    """
    기본 예측(Base Forecast)에 대한 성능 평가 지표를 출력하는 함수.

    각 사이트별로 MLR, SVR, LGB, MLP 모델에 대해
    MAE, MSE, RMSE를 계산하고 콘솔에 출력한다.

    Args:
        filtered_df_list (list): 운전 시간대로 필터링된 DataFrame 리스트.
        site_names (list): 각 DataFrame에 대응되는 사이트 이름 리스트.

    Returns:
        None
    """
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


def do_forecast_and_evaluate(path: str, features: list, target: str, shifted_target: str, test_size: float,
                             origin_df_list: list, site_names: list, operation_hours_array: list) -> list:
    """
    전체 base forecast 파이프라인을 실행하고 평가까지 수행하는 함수.

    처리 내용:
        1. do_base_forecast: 기본 예측 수행 및 오차 계산.
        2. save_forecast_result: base forecast 결과를 CSV로 저장.
        3. filter_by_operation_hours: 운전 시간대에 해당하는 시점만 필터링.
        4. print_out_forecast_eval_result: MAE, MSE, RMSE 출력.

    Args:
        path (str): 데이터 및 결과 저장을 위한 상위 디렉터리 경로.
        features (list): base forecast에서 사용할 피처 리스트.
        target (str): 예측 타깃 컬럼명.
        shifted_target (str): 다음 시점 타깃을 저장할 컬럼명.
        test_size (float): 테스트 비율.
        origin_df_list (list): 전처리 완료된 원본 DataFrame 리스트.
        site_names (list): 사이트 이름 리스트.
        operation_hours_array (list): 운전 시간대 정보 리스트.

    Returns:
        list: 운전 시간대 기준으로 필터링된 결과 DataFrame 리스트.
    """
    result_for_reforecast_df_list = do_base_forecast(origin_df_list, site_names, features, target, shifted_target,
                                                     test_size, path)
    save_forecast_result(result_for_reforecast_df_list, site_names, path)
    filtered_df_list = filter_by_operation_hours(result_for_reforecast_df_list, operation_hours_array, test_size)
    print_out_forecast_eval_result(filtered_df_list, site_names)

    return filtered_df_list


# ## Reforecast
def create_reforecast_features_and_targets() -> (list, list, list):
    """
    재예측(Reforecast)에 필요한 피처 리스트와 타깃/시프트 타깃 리스트를 생성하는 함수.

    Returns:
        tuple[list, list, list]:
            - features_list: MLR, SVR, LGB, MLP 각각에 대한 피처 리스트.
            - targets: 각 모델의 오차 컬럼명 리스트 (예: 'error_mlr', 'error_svr', ...).
            - shifted_targets: 한 스텝 뒤로 shift된 오차 컬럼명 리스트.
    """
    features_list = [forecast_mlr_features, forecast_svr_features, forecast_lgb_features, forecast_mlp_features]
    mlr_target = 'error_mlr'
    svr_target = 'error_svr'
    lgb_target = 'error_lgb'
    mlp_target = 'error_mlp'
    targets = [mlr_target, svr_target, lgb_target, mlp_target]

    # shift된 오차 타겟명
    shifted_mlr_target = 'error_mlr_shifted'
    shifted_svr_target = 'error_svr_shifted'
    shifted_lgb_target = 'error_lgb_shifted'
    shifted_mlp_target = 'error_mlp_shifted'
    shifted_targets = [shifted_mlr_target, shifted_svr_target, shifted_lgb_target, shifted_mlp_target]

    return features_list, targets, shifted_targets


def do_reforecast(targets: list, shifted_targets: list, features_list: list, site_names: list,
                  result_for_reforecast_df_list: list, test_size: float, path: str):
    """
    base forecast에서 얻은 오차를 재예측(Reforecast)하여
    보정된 PV 발전량을 계산하고 저장하는 함수.

    처리 내용:
        1. 각 forecasting 모델(mlr, svr, lgb, mlp)에 대해:
            - 오차 타깃(error_*)을 한 스텝 뒤로 shift하여 shifted_target에 저장.
            - 해당 오차를 예측하기 위한 재예측 모델(MLR, SVR, LGB, MLP) 학습.
            - 예측된 오차(pred_error_*)를 바탕으로 보정된 PV(reforecasted_PV_*) 계산.
            - 결과를 CSV 파일로 저장.

    Args:
        targets (list): 오차 타깃 컬럼명 리스트.
        shifted_targets (list): shift된 오차 타깃 컬럼명 리스트.
        features_list (list): 재예측에 사용할 피처 리스트들.
        site_names (list): 사이트 이름 리스트.
        result_for_reforecast_df_list (list): base forecast 결과 DataFrame 리스트.
        test_size (float): 테스트 비율.
        path (str): 모델 및 결과 저장 경로.

    Returns:
        list: 각 forecast 모델별 최종 재예측 결과 DataFrame 리스트 목록 (길이 4의 리스트).
    """
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

            # 오차 타깃을 한 스텝 뒤로 shift하여 다음 시점의 오차를 예측하도록 구성
            df[shifted_target] = df[target].shift(-1)
            df = df.dropna().reset_index(drop=True)

            # 실제 존재하는 피처만 사용
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

            # 모델 학습 (오차 재예측 모델)
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

            # 예측된 오차를 t+1 시점에 대응시키기 위해 한 스텝 shift
            for col in ['pred_error_mlr', 'pred_error_svr', 'pred_error_lgb', 'pred_error_mlp']:
                df_result[col] = df_result[col].shift(1)

            # 시간 복원
            X_test_result = df_result.copy()
            X_test_result.loc[:split_index - 1, 'Time'] = time_train.values
            X_test_result.loc[split_index:, 'Time'] = time_test.values
            X_test_result.dropna(subset=['pred_error_mlr', 'pred_error_svr', 'pred_error_lgb', 'pred_error_mlp'],
                                 inplace=True)

            reforecasted_df_list.append(X_test_result)

            # 재예측 모델 저장
            joblib.dump(mlr,
                        f"{path}/result_of_paper/reforecast/mlr_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            joblib.dump(svr,
                        f"{path}/result_of_paper/reforecast/svr_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            joblib.dump(mlp,
                        f"{path}/result_of_paper/reforecast/mlp_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            lgb_model.booster_.save_model(
                f"{path}/result_of_paper/reforecast/lgb_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.txt")

        # 재예측 값 계산 및 저장 (예측값 - 예측된 오차)
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

        # 각 사이트별 최종 재예측 결과를 CSV로 저장
        for final_result_df, df_index in zip(final_result_list, site_names):
            final_result_df.to_csv(f"{path}/result_of_paper/reforecast/{forecast_model}_{df_index}_.csv",
                                   index=False)

    return final_result_lists


def print_out_reforecast_eval(final_result_lists: list, operation_hours_array: list, site_names: list,
                              test_size: float):
    """
    재예측(Reforecast) 결과에 대한 성능 평가 지표를 출력하는 함수.

    각 forecast 모델(mlr, svr, lgb, mlp)에 대해:
        - 운전 시간대만 필터링
        - 재예측된 PV(reforecasted_PV_*)와 실제 Power (MW) 간의
          MAE, MSE, RMSE를 계산하여 콘솔에 출력.

    Args:
        final_result_lists (list): do_reforecast에서 생성된 최종 재예측 DataFrame 리스트 목록.
        operation_hours_array (list): 각 사이트별 운전 시간대 정보 리스트.
        site_names (list): 사이트 이름 리스트.
        test_size (float): 테스트 비율 (시계열 분할 기준 유지용).

    Returns:
        None
    """
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

            # 현재 구조상, filtered_df_list가 점점 쌓이는 형태이므로
            # 매 루프마다 이미 쌓인 모든 사이트에 대해 평가를 반복하게 됨
            # (중복 출력이 발생할 수 있으나, 로직은 그대로 유지함)
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


def do_reforecast_and_evaluate(targets: list, shifted_targets: list, features_list: list, site_names: list,
                               result_for_reforecast_df_list: list, operation_hours_array: list, test_size: float,
                               path: str):
    """
    전체 재예측(Reforecast) 및 평가 파이프라인을 실행하는 함수.

    처리 내용:
        1. do_reforecast: base forecast 오차에 대한 재예측 수행 및 보정된 PV 산출.
        2. print_out_reforecast_eval: 운전 시간대 기준 필터링 후 성능 평가 출력.

    Args:
        targets (list): 오차 타깃 컬럼명 리스트.
        shifted_targets (list): shift된 오차 타깃 컬럼명 리스트.
        features_list (list): 재예측 피처 리스트.
        site_names (list): 사이트 이름 리스트.
        result_for_reforecast_df_list (list): base forecast 결과 DataFrame 리스트.
        operation_hours_array (list): 각 사이트별 운전 시간대 리스트.
        test_size (float): 테스트 비율.
        path (str): 결과 저장 경로.

    Returns:
        None
    """
    final_result_lists = do_reforecast(targets, shifted_targets, features_list, site_names,
                                       result_for_reforecast_df_list, test_size, path)
    print_out_reforecast_eval(final_result_lists, operation_hours_array, site_names, test_size)
