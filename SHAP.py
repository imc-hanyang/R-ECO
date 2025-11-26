#%%
"""
재예측(Reforecast) 및 기본 예측(Base Forecast)에 대해
SHAP 기반 피처 중요도를 계산하고 시각화하는 유틸리티 모듈.

주요 기능:
- 재예측 모델(MLR, SVR, LGB, MLP)에 대한 SHAP/Permutation 중요도 계산 및 시각화
- 기본 예측 모델(MLR, SVR, LGB, MLP)에 대한 SHAP/Permutation 중요도 계산 및 시각화
- Kneedle-유사 방식(거리 기반)으로 elbow point를 찾고, 핵심 피처 식별
"""

import os
import pandas as pd
import shap
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from feature_lists import forecast_features
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import multiprocessing


def compute_feature_importance(f, X, base_pred, model, n_repeats, rng):
    """
    하나의 피처 f에 대해 permutation importance를 반복 계산하는 함수.

    Args:
        f: 중요도를 계산할 단일 피처 이름(컬럼명).
        X (DataFrame): 피처 데이터셋.
        base_pred (ndarray): 원본 X에 대한 모델 예측값.
        model: predict 메서드를 가진 학습된 모델 객체.
        n_repeats (int): permutation을 반복할 횟수.
        rng (np.random.RandomState): 난수 생성기(재현성 보장용).

    Returns:
        tuple: (피처 이름 f, 각 샘플별 중요도 벡터의 평균값 ndarray)
    """
    feature_diffs = []
    for _ in range(n_repeats):
        X_permuted = X.copy()
        # 피처 f의 값을 랜덤하게 섞어서(permutation) 모델 예측 변화 관찰
        X_permuted[f] = rng.permutation(X_permuted[f].values)
        pred_permuted = model.predict(X_permuted.values)
        diff = np.abs(base_pred - pred_permuted)
        feature_diffs.append(diff)
    return f, np.mean(np.stack(feature_diffs), axis=0)


def permutation_importance_per_instance(columns, model, X, n_repeats=5, random_state=42, n_jobs=-1):
    """
    여러 피처에 대해 샘플 단위 permutation importance를 계산하는 함수.

    Args:
        columns (list): 중요도를 계산할 피처(컬럼) 이름 리스트.
        model: predict 메서드를 가진 학습된 모델 객체.
        X (DataFrame): 피처 데이터셋.
        n_repeats (int, optional): 각 피처에 대해 permutation 반복 횟수.
        random_state (int, optional): 난수 시드.
        n_jobs (int, optional): 병렬 실행 시 사용할 CPU 코어 수(-1이면 전체).

    Returns:
        dict: {피처명: 샘플별 중요도 벡터} 형태의 딕셔너리.
    """
    rng = np.random.RandomState(random_state)
    X = X.copy()
    base_pred = model.predict(X.values)
    # 각 피처에 대해 compute_feature_importance를 병렬로 수행
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_feature_importance)(f, X, base_pred, model, n_repeats, rng)
        for f in columns
    )
    return {f: v for f, v in results}


#%%
def do_reforecast_train_shap(path: str, reforecast_models: list, forecast_models: list, test_size: float,
                             site_names: list, target: str, shifted_target: str):
    """
    재예측(Reforecast) 단계에서 학습된 모델들에 대해 SHAP/Permutation 기반
    피처 중요도를 계산하고 시각화하는 함수.

    처리 개요:
        1. 재예측 결과 CSV(mlr/svr/lgb/mlp)를 읽어와서 DataFrame 리스트로 구성.
        2. forecast_model(mlr/svr/lgb/mlp) 별로, 각 사이트(site)에 대해 반복.
        3. 재예측 모델(reforecast_mlr/svr/lgb/mlp)을 로드.
        4. 각 reforecast_model에 대해:
            - MLR/LGB/MLP: SHAP 기반 중요도 계산 및 summary plot.
            - SVR: permutation_importance_per_instance 기반 중요도 계산.
        5. 거리 기반 elbow point를 계산하여 핵심 피처를 시각적으로 강조.

    Args:
        path (str): 결과 파일이 저장된 상위 디렉터리 경로.
        reforecast_models (list): ['mlr', 'svr', 'lgb', 'mlp'] 등 재예측 모델 종류 리스트.
        forecast_models (list): ['mlr', 'svr', 'lgb', 'mlp'] 등 base forecast 모델 종류 리스트.
        test_size (float): 학습/테스트 분할에서 사용할 테스트 비율.
        site_names (list): 사이트(발전소) 이름 또는 ID 리스트.
        target (str): 오차 타깃 컬럼명 (예: 'error_mlr', 'error_svr' 등).
        shifted_target (str): shift된 오차 타깃 컬럼명 (예: 'error_mlr_shifted' 등).

    Returns:
        None
    """
    features = forecast_features.copy()
    mlr_files = sorted([os.path.join(path+'/result_of_paper/reforecast/', file) for file in os.listdir(path+'/result_of_paper/reforecast/') if file.endswith('.csv') and 'mlr' in file])
    svr_files = sorted([os.path.join(path+'/result_of_paper/reforecast/', file) for file in os.listdir(path+'/result_of_paper/reforecast/') if file.endswith('.csv') and 'svr' in file])
    lgb_files = sorted([os.path.join(path+'/result_of_paper/reforecast/', file) for file in os.listdir(path+'/result_of_paper/reforecast/') if file.endswith('.csv') and 'lgb' in file])
    mlp_files = sorted([os.path.join(path+'/result_of_paper/reforecast/', file) for file in os.listdir(path+'/result_of_paper/reforecast/') if file.endswith('.csv') and 'mlp' in file])

    mlr_df_list = []
    svr_df_list = []
    lgb_df_list = []
    mlp_df_list = []

    # 재예측 결과 CSV들을 DataFrame 리스트로 로드
    for file in mlr_files:
      mlr_df_list.append(pd.read_csv(file))
    for file in svr_files:
      svr_df_list.append(pd.read_csv(file))
    for file in lgb_files:
      lgb_df_list.append(pd.read_csv(file))
    for file in mlp_files:
      mlp_df_list.append(pd.read_csv(file))

    df_lists=[mlr_df_list, svr_df_list,lgb_df_list, mlp_df_list]

    # forecast_model(mlr/svr/lgb/mlp) 별로 루프
    for forecast_model, df_list in zip(forecast_models, df_lists):
      print('==============================================================================================')
      print(f'forecast_model : {forecast_model.upper()}')

      # 각 사이트별 루프
      for df, site in zip(df_list, site_names):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'site : {site}')

        # 재예측 모델 종류별 루프 (mlr/svr/lgb/mlp)
        for reforecast_model in reforecast_models:
          print('--------------------------------------------------------------------------------------------------------------------------------')

          df_copy = df.copy()
          # 다음 시점 오차 예측을 위한 shift 적용
          df_copy[shifted_target] = df_copy[target].shift(-1)
          df_copy = df_copy.dropna().reset_index(drop=True)

          split_index = int(len(df_copy) * (1 - test_size))

          # base forecast 피처 + 해당 forecast_model의 오차/예측값 추가
          added_features = features.copy() + [f'error_{forecast_model}', f'pred_{forecast_model}']
          added_features = [col for col in added_features if col in df_copy.columns]
          X = df_copy[added_features]
          y = df_copy[shifted_target]
          X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
          y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

          # 스케일링(표준화)
          scaler = StandardScaler()
          X_train_scaled = scaler.fit_transform(X_train)
          X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
          X_test_scaled = scaler.transform(X_test)
          X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

          # 재예측 모델 로드
          reforecast_mlr = joblib.load(f"{path}/result_of_paper/reforecast/mlr_reforecasting_model_{forecast_model}_forecast_{str(site)}.joblib")
          reforecast_svr = joblib.load(f"{path}/result_of_paper/reforecast/svr_reforecasting_model_{forecast_model}_forecast_{str(site)}.joblib")
          reforecast_lgb_model = lgb.Booster(model_file=f"{path}/result_of_paper/reforecast/lgb_reforecasting_model_{forecast_model}_forecast_{str(site)}.txt")
          reforecast_mlp = joblib.load(f"{path}/result_of_paper/reforecast/mlp_reforecasting_model_{forecast_model}_forecast_{site}.joblib")


          if reforecast_model == 'mlr':
              # ------------------- MLR -------------------
              # LinearRegression 재예측 모델에 대한 SHAP 값 계산
              explainer_mlr = shap.Explainer(reforecast_mlr, X_train_scaled_df)
              shap_values_mlr = explainer_mlr(X_train_scaled_df)

              # 피처별 |SHAP|의 평균값을 중요도로 사용
              shap_importance = np.abs(shap_values_mlr.values).mean(axis=0)

              shap_df = pd.DataFrame({
                  'feature': X_train.columns,
                  'mean_abs_shap': shap_importance
              }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

              print("\n[MLR Feature Importance (SHAP)]")
              for _, row in shap_df.iterrows():
                print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

              # elbow point (Kneedle 유사 방식) 계산
              x = np.arange(len(shap_df))
              y = shap_df['mean_abs_shap'].values
              x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
              y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

              start = np.array([x_scaled[0], y_scaled[0]])
              end = np.array([x_scaled[-1], y_scaled[-1]])

              line_vec = end - start
              line_vec_norm = line_vec / np.linalg.norm(line_vec)
              point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
              projections = np.dot(point_vecs, line_vec_norm)
              projected_points = np.outer(projections, line_vec_norm) + start
              distances = np.linalg.norm(point_vecs - projected_points, axis=1)

              elbow_idx = np.argmax(distances)
              elbow_feature = shap_df.loc[elbow_idx, 'feature']
              elbow_value = shap_df.loc[elbow_idx, 'mean_abs_shap']

              # SHAP 중요도 + elbow point 시각화
              plt.figure(figsize=(10, 6))
              plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
              plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
              plt.title("MLR SHAP Feature Importance with Elbow Point")
              plt.xlabel("Feature Rank")
              plt.ylabel("Mean |SHAP value|")
              plt.legend()
              plt.grid(True)
              plt.show()

              # SHAP summary plot (dot)
              shap.summary_plot(shap_values_mlr, X_train_scaled_df, plot_type="dot", show=True)


          elif reforecast_model == 'svr':
              # ------------------- SVR -------------------
              # SVR은 SHAP 기본 Explainer가 무겁기 때문에 permutation 기반 중요도 사용
              X_sample_df, y_sample = resample(X_train_scaled_df, y_train, n_samples=5000, random_state=42)
              importance_dict = permutation_importance_per_instance(
                  columns=X_sample_df.columns,
                  model=reforecast_svr,
                  X=X_sample_df,
                  n_repeats=5,
                  n_jobs=-1  # 모든 CPU 코어 사용
              )

              # ▶ 평균 중요도로 정리 (SHAP 스타일 표기)
              shap_style_df = pd.DataFrame({
                  'feature': X_sample_df.columns,
                  'mean_abs_shap': [np.mean(np.abs(v)) for v in importance_dict.values()]
              }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

              # ▶ 출력
              for _, row in shap_style_df.iterrows():
                  print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

              # elbow point 계산 (거리 기반)
              x = np.arange(len(shap_style_df))
              y = shap_style_df['mean_abs_shap'].values
              x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
              y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

              start = np.array([x_scaled[0], y_scaled[0]])
              end = np.array([x_scaled[-1], y_scaled[-1]])

              line_vec = end - start
              line_vec_norm = line_vec / np.linalg.norm(line_vec)
              point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
              projections = np.dot(point_vecs, line_vec_norm)
              projected_points = np.outer(projections, line_vec_norm) + start
              distances = np.linalg.norm(point_vecs - projected_points, axis=1)

              elbow_idx = np.argmax(distances)
              elbow_feature = shap_style_df.loc[elbow_idx, 'feature']
              elbow_value = shap_style_df.loc[elbow_idx, 'mean_abs_shap']

              plt.figure(figsize=(10, 6))
              plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
              plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
              plt.title("SVR SHAP Feature Importance with Elbow Point")
              plt.xlabel("Feature Rank")
              plt.ylabel("Mean |SHAP value|")
              plt.legend()
              plt.grid(True)
              plt.show()

              # permutation importance를 SHAP summary plot 스타일로 시각화하기 위한 배열 구성
              svr_array = np.stack([v for v in importance_dict.values()], axis=-1)
              shap.summary_plot(svr_array, X_sample_df, plot_type="dot", show=True)


          elif reforecast_model == 'lgb':
          # ------------------- LGB -------------------explainer_lgb = shap.Explainer(reforecast_lgb_model)

            # LightGBM Booster에 대한 SHAP 값 계산
            explainer_lgb = shap.Explainer(reforecast_lgb_model)
            shap_values_lgb = explainer_lgb(X_train)

            shap_importance = np.abs(shap_values_lgb.values).mean(axis=0)

            # DataFrame으로 정리 (피처별 평균 |SHAP|)
            shap_df = pd.DataFrame({
                'feature': X_test.columns,
                'mean_abs_shap': shap_importance
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

            # 출력
            print("\n[LGB Feature Importance (SHAP)]")
            for _, row in shap_df.iterrows():
                print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

            # kneedle 유사: 직선과의 거리 기반으로 elbow point 계산
            x = np.arange(len(shap_df))
            y = shap_df['mean_abs_shap'].values
            x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
            y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

            # 직선 시작점과 끝점
            start = np.array([x_scaled[0], y_scaled[0]])
            end = np.array([x_scaled[-1], y_scaled[-1]])

            # 각 점에서 직선까지의 수직 거리 계산
            line_vec = end - start
            line_vec_norm = line_vec / np.linalg.norm(line_vec)
            point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
            projections = np.dot(point_vecs, line_vec_norm)
            projected_points = np.outer(projections, line_vec_norm) + start
            distances = np.linalg.norm(point_vecs - projected_points, axis=1)

            # 가장 멀리 떨어진 점이 elbow point
            elbow_idx = np.argmax(distances)
            elbow_feature = shap_df.loc[elbow_idx, 'feature']
            elbow_value = shap_df.loc[elbow_idx, 'mean_abs_shap']

            # 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
            plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
            plt.title("SHAP Feature Importance with Elbow Point (Distance Method)")
            plt.xlabel("Feature Rank")
            plt.ylabel("Mean |SHAP value|")
            plt.legend()
            plt.grid(True)
            plt.show()

            # SHAP summary plot
            shap.summary_plot(shap_values_lgb, X_train, plot_type="dot", show=True)


          elif reforecast_model == 'mlp':
            # ------------------- MLP -------------------
            # KernelExplainer + multiprocessing을 이용한 SHAP 계산 (비용이 매우 큰 작업)
            def compute_shap_partial(batch):
                explainer = shap.KernelExplainer(reforecast_mlp.predict, background)
                return explainer.shap_values(batch)

            # 대상/배경 샘플 추출
            X_target = shap.utils.sample(X_train_scaled_df, 2000, random_state=42)
            background = shap.utils.sample(X_target, 100, random_state=42)
            X_batches = np.array_split(X_target, 8)
            with multiprocessing.Pool(processes=8) as pool:
                shap_parts = pool.map(compute_shap_partial, X_batches)
            shap_values = np.concatenate(shap_parts, axis=0)

            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': X_train_scaled_df.columns,
                'mean_abs_shap': shap_importance
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

            print("\n[MLP Feature Importance (SHAP)]")
            for _, row in shap_df.iterrows():
                print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

            # elbow point 계산 및 시각화
            x = np.arange(len(shap_df))
            y = shap_df['mean_abs_shap'].values
            x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
            y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()
            start = np.array([x_scaled[0], y_scaled[0]])
            end = np.array([x_scaled[-1], y_scaled[-1]])
            line_vec = end - start
            line_vec_norm = line_vec / np.linalg.norm(line_vec)
            point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
            projections = np.dot(point_vecs, line_vec_norm)
            projected_points = np.outer(projections, line_vec_norm) + start
            distances = np.linalg.norm(point_vecs - projected_points, axis=1)
            elbow_idx = np.argmax(distances)
            elbow_feature = shap_df.loc[elbow_idx, 'feature']
            elbow_value = shap_df.loc[elbow_idx, 'mean_abs_shap']

            plt.figure(figsize=(10, 6))
            plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
            plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
            plt.title("MLP SHAP Feature Importance with Elbow Point")
            plt.xlabel("Feature Rank")
            plt.ylabel("Mean |SHAP value|")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # SHAP summary plot 출력 (반환값은 None이라 print(None)이 나올 수 있음)
            print(shap.summary_plot(shap_values, X_target, plot_type="dot", show=True))


def do_forecast_train_shap(path: str, forecast_models: list, test_size: float, site_names: list,
                           target: str, shifted_target: str):
    """
    base forecast 단계에서 학습된 모델들에 대해 SHAP/Permutation 기반
    피처 중요도를 계산하고 시각화하는 함수.

    처리 개요:
        1. base forecast 결과 CSV(현재는 'mlr' 포함 파일만 사용)를 읽어 DataFrame 리스트 구성.
        2. 각 사이트(site)에 대해:
            - shift된 타깃(shifted_target)을 생성.
            - 스케일링 후 MLR/SVR/LGB/MLP 모델을 로드.
        3. forecast_model에 따라:
            - MLR/LGB/MLP: SHAP 기반 중요도 계산.
            - SVR: permutation_importance_per_instance 기반 중요도 계산.
        4. elbow point를 계산하여 주요 피처를 시각화.

    Args:
        path (str): 결과 파일이 저장된 상위 디렉터리 경로.
        forecast_models (list): ['mlr', 'svr', 'lgb', 'mlp'] 등 예측 모델 종류 리스트.
        test_size (float): 학습/테스트 분할에서 사용할 테스트 비율.
        site_names (list): 사이트 이름 또는 ID 리스트.
        target (str): 예측 타깃(혹은 오차) 컬럼명.
        shifted_target (str): shift된 타깃 컬럼명.

    Returns:
        None
    """
    files = sorted([os.path.join(path+'/result_of_paper/forecast/', file) for file in os.listdir(path+'/result_of_paper/forecast/') if file.endswith('.csv')])

    df_list = []
    for file in files:
      df_list.append(pd.read_csv(file))

    # 사이트별로 base forecast SHAP 분석
    for df, site in zip(df_list, site_names):
      print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print(f'site : {site}')

      df_copy = df.copy()
      # 다음 시점 예측을 위한 shift
      df_copy[shifted_target] = df_copy[target].shift(-1)
      df_copy = df_copy.dropna().reset_index(drop=True)

      split_index = int(len(df_copy) * (1 - test_size))

      X = df_copy[forecast_features]
      y = df_copy[shifted_target]
      X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
      y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

      # 스케일링
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
      X_test_scaled = scaler.transform(X_test)
      X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

      # base forecast 모델 로드
      forecast_mlr = joblib.load(f"{path}/result_of_paper/forecast/mlr_forecasting_model_{str(site)}.joblib")
      forecast_svr = joblib.load(f"{path}/result_of_paper/forecast/svr_forecasting_model_{str(site)}.joblib")
      forecast_lgb_model = lgb.Booster(model_file=f"{path}/result_of_paper/forecast/lgb_forecasting_model_{str(site)}.txt")
      forecast_mlp = joblib.load(f"{path}/result_of_paper/forecast/mlp_forecasting_model_{str(site)}.joblib")

      # 모델 종류별 SHAP/Permutation 분석
      for forecast_model in forecast_models:
        print('--------------------------------------------------------------------------------------------------------------------------------')
        print(f'forecast_model : {forecast_model}')

        if forecast_model == 'mlr':
            # ------------------- MLR -------------------
            explainer_mlr = shap.Explainer(forecast_mlr, X_train_scaled_df)
            shap_values_mlr = explainer_mlr(X_train_scaled_df)

            shap_importance = np.abs(shap_values_mlr.values).mean(axis=0)

            shap_df = pd.DataFrame({
                'feature': X_train.columns,
                'mean_abs_shap': shap_importance
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

            print("\n[MLR Feature Importance (SHAP)]")
            for _, row in shap_df.iterrows():
              print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

            # elbow point 계산
            x = np.arange(len(shap_df))
            y = shap_df['mean_abs_shap'].values
            x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
            y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

            start = np.array([x_scaled[0], y_scaled[0]])
            end = np.array([x_scaled[-1], y_scaled[-1]])

            line_vec = end - start
            line_vec_norm = line_vec / np.linalg.norm(line_vec)
            point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
            projections = np.dot(point_vecs, line_vec_norm)
            projected_points = np.outer(projections, line_vec_norm) + start
            distances = np.linalg.norm(point_vecs - projected_points, axis=1)

            elbow_idx = np.argmax(distances)
            elbow_feature = shap_df.loc[elbow_idx, 'feature']
            elbow_value = shap_df.loc[elbow_idx, 'mean_abs_shap']

            plt.figure(figsize=(10, 6))
            plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
            plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
            plt.title("MLR SHAP Feature Importance with Elbow Point")
            plt.xlabel("Feature Rank")
            plt.ylabel("Mean |SHAP value|")
            plt.legend()
            plt.grid(True)
            plt.show()

        elif forecast_model == 'svr':
            # ------------------- SVR -------------------
            # SVR에 대해 permutation 기반 중요도 계산
            X_sample_df, y_sample = resample(X_train_scaled_df, y_train, n_samples=5000, random_state=42)
            importance_dict = permutation_importance_per_instance(
                columns=X_sample_df.columns,
                model=forecast_svr,
                X=X_sample_df,
                n_repeats=5,
                n_jobs=-1  # 모든 CPU 코어 사용
            )

            # ▶ 평균 중요도로 정리
            shap_style_df = pd.DataFrame({
                'feature': X_sample_df.columns,
                'mean_abs_shap': [np.mean(np.abs(v)) for v in importance_dict.values()]
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

            # ▶ 출력
            for _, row in shap_style_df.iterrows():
                print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

            # elbow point 계산
            x = np.arange(len(shap_style_df))
            y = shap_style_df['mean_abs_shap'].values
            x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
            y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

            start = np.array([x_scaled[0], y_scaled[0]])
            end = np.array([x_scaled[-1], y_scaled[-1]])

            line_vec = end - start
            line_vec_norm = line_vec / np.linalg.norm(line_vec)
            point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
            projections = np.dot(point_vecs, line_vec_norm)
            projected_points = np.outer(projections, line_vec_norm) + start
            distances = np.linalg.norm(point_vecs - projected_points, axis=1)

            elbow_idx = np.argmax(distances)
            elbow_feature = shap_style_df.loc[elbow_idx, 'feature']
            elbow_value = shap_style_df.loc[elbow_idx, 'mean_abs_shap']

            plt.figure(figsize=(10, 6))
            plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
            plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
            plt.title("SVR SHAP Feature Importance with Elbow Point")
            plt.xlabel("Feature Rank")
            plt.ylabel("Mean |SHAP value|")
            plt.legend()
            plt.grid(True)
            plt.show()


        elif forecast_model == 'lgb':
        # ------------------- LGB -------------------explainer_lgb = shap.Explainer(reforecast_lgb_model)

          # LightGBM Booster에 대한 SHAP 값 계산
          explainer_lgb = shap.Explainer(forecast_lgb_model)
          shap_values_lgb = explainer_lgb(X_train)

          shap_importance = np.abs(shap_values_lgb.values).mean(axis=0)

          # DataFrame으로 정리 (피처별 평균 |SHAP|)
          shap_df = pd.DataFrame({
              'feature': X_test.columns,
              'mean_abs_shap': shap_importance
          }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

          # 출력
          print("\n[LGB Feature Importance (SHAP)]")
          for _, row in shap_df.iterrows():
              print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

          # kneedle 유사: 직선과의 거리 기반으로 elbow point 계산
          x = np.arange(len(shap_df))
          y = shap_df['mean_abs_shap'].values
          x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
          y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

          # 직선 시작점과 끝점
          start = np.array([x_scaled[0], y_scaled[0]])
          end = np.array([x_scaled[-1], y_scaled[-1]])

          # 각 점에서 직선까지의 수직 거리 계산
          line_vec = end - start
          line_vec_norm = line_vec / np.linalg.norm(line_vec)
          point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
          projections = np.dot(point_vecs, line_vec_norm)
          projected_points = np.outer(projections, line_vec_norm) + start
          distances = np.linalg.norm(point_vecs - projected_points, axis=1)

          # 가장 멀리 떨어진 점이 elbow point
          elbow_idx = np.argmax(distances)
          elbow_feature = shap_df.loc[elbow_idx, 'feature']
          elbow_value = shap_df.loc[elbow_idx, 'mean_abs_shap']

          # 시각화
          plt.figure(figsize=(10, 6))
          plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
          plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
          plt.title("SHAP Feature Importance with Elbow Point (Distance Method)")
          plt.xlabel("Feature Rank")
          plt.ylabel("Mean |SHAP value|")
          plt.legend()
          plt.grid(True)
          plt.show()

        elif forecast_model == 'mlp':
          # ------------------- MLP -------------------
          # KernelExplainer + multiprocessing을 이용한 SHAP 계산 (비용이 매우 큰 작업)
          def compute_shap_partial(batch):
              explainer = shap.KernelExplainer(forecast_mlp.predict, background)
              return explainer.shap_values(batch)

          X_target = shap.utils.sample(X_train_scaled_df, 2000, random_state=42)
          background = shap.utils.sample(X_target, 100, random_state=42)
          X_batches = np.array_split(X_target, 8)
          with multiprocessing.Pool(processes=8) as pool:
              shap_parts = pool.map(compute_shap_partial, X_batches)
          shap_values_mlp = np.concatenate(shap_parts, axis=0)

          shap_importance = np.abs(shap_values_mlp).mean(axis=0)
          shap_df = pd.DataFrame({
              'feature': X_train_scaled_df.columns,
              'mean_abs_shap': shap_importance
          }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

          print("\n[MLP Feature Importance (SHAP)]")
          for _, row in shap_df.iterrows():
              print(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

          # elbow point 계산 및 시각화
          x = np.arange(len(shap_df))
          y = shap_df['mean_abs_shap'].values
          x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
          y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

          start = np.array([x_scaled[0], y_scaled[0]])
          end = np.array([x_scaled[-1], y_scaled[-1]])
          line_vec = end - start
          line_vec_norm = line_vec / np.linalg.norm(line_vec)
          point_vecs = np.stack([x_scaled - start[0], y_scaled - start[1]], axis=1)
          projections = np.dot(point_vecs, line_vec_norm)
          projected_points = np.outer(projections, line_vec_norm) + start
          distances = np.linalg.norm(point_vecs - projected_points, axis=1)

          elbow_idx = np.argmax(distances)
          elbow_feature = shap_df.loc[elbow_idx, 'feature']
          elbow_value = shap_df.loc[elbow_idx, 'mean_abs_shap']

          plt.figure(figsize=(10, 6))
          plt.plot(x, y, marker='o', label='SHAP importance', linewidth=2)
          plt.scatter(elbow_idx, elbow_value, color='red', marker='*', s=300, label=f'Elbow Point: {elbow_feature}')
          plt.title("MLP SHAP Feature Importance with Elbow Point")
          plt.xlabel("Feature Rank")
          plt.ylabel("Mean |SHAP value|")
          plt.legend()
          plt.grid(True)
          plt.show()
