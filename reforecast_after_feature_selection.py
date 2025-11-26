"""
사이트별·모델별로 선택된 재예측(reforecast) 피처를 사용하여
오차 재예측 모델(MLR, SVR, LGB, MLP)을 학습하고,
재예측된 PV 출력(reforecasted_PV_*)을 계산·평가하는 모듈.

주요 기능:
- forecast 모델별(MLR/SVR/LGB/MLP) 재예측용 피처 리스트 정의
- reforecast 단계 입력 CSV 로딩(get_files_in_directory)
- 오차 재예측 모델 학습 및 예측(do_reforecast)
- 운전 시간대별 성능 평가(do_evaluate)
- 전체 파이프라인 실행 및 결과 저장(do_reforecast_and_evaluate_and_save)
"""

import os
import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
import numpy as np
from datetime import time
from sklearn.neural_network import MLPRegressor

# =====================================================================================
# 1) MLR forecast에 대한 reforecast feature 리스트 (사이트별)
#    - siteX_features_mlr_reforecast: MLR 예측 오차(error_mlr)를 재예측하기 위한 입력 피처들
# =====================================================================================

# mlr forcecast reforecast features
site1_features_mlr_reforecast = ['same_time_max', 'lag_8', 'diff_15min', 'error_mlr', 'same_weekday_mean', 'hour_sin',
                                 'hour_cos', 'same_weekday_std', 'same_time_mean', 'lag_1']
site2_features_mlr_reforecast = ['diff_15min', 'error_mlr', 'lag_8', 'hour_cos', 'same_time_mean', 'pred_mlr', 'lag_1',
                                 'same_time_max', 'Air temperature  (°C)', 'lag_4']
site4_features_mlr_reforecast = ['error_mlr', 'diff_15min', 'lag_8', 'pred_mlr', 'same_time_mean', 'lag_1',
                                 'diff_30min', 'same_weekday_mean']
site5_features_mlr_reforecast = ['error_mlr', 'diff_15min', 'lag_8', 'pred_mlr', 'lag_1', 'same_time_mean',
                                 'diff_30min', 'same_weekday_mean']
site6_features_mlr_reforecast = ['same_time_max', 'same_time_mean', 'diff_15min', 'error_mlr', 'hour_sin', 'pred_mlr']
site7_features_mlr_reforecast = ['error_mlr', 'diff_15min', 'same_time_max', 'Total solar irradiance (W/m2)',
                                 'hour_sin', 'pred_mlr', 'lag_8', 'same_weekday_mean', 'lag_1', 'same_time_mean',
                                 'hour_cos', 'same_weekday_std', 'diff_45min', 'month_sin', 'lag_3']
site8_features_mlr_reforecast = ['error_mlr', 'diff_15min', 'same_time_max', 'diff_30min', 'same_weekday_std',
                                 'hour_cos', 'same_time_mean', 'same_weekday_mean', 'lag_8',
                                 'Total solar irradiance (W/m2)']

site1_features_svr_reforecast = ['Total solar irradiance (W/m2)', 'diff_15min', 'lag_8', 'same_time_max', 'hour_sin',
                                 'hour_cos', 'error_mlr', 'same_time_mean', 'same_weekday_mean', 'lag_1']
site2_features_svr_reforecast = ['hour_cos', 'diff_15min', 'error_mlr', 'same_time_mean', 'hour_sin',
                                 'Total solar irradiance (W/m2)', 'lag_8', 'same_time_max', 'diff_60min']
site4_features_svr_reforecast = ['hour_sin', 'lag_8', 'error_mlr', 'same_weekday_mean', 'same_weekday_std',
                                 'diff_15min']
site5_features_svr_reforecast = ['hour_sin', 'error_mlr', 'Total solar irradiance (W/m2)', 'lag_8', 'same_weekday_mean',
                                 'same_weekday_std', 'Atmosphere (hpa)']
site6_features_svr_reforecast = ['same_time_mean', 'Total solar irradiance (W/m2)', 'lag_8', 'hour_sin',
                                 'same_time_max', 'same_weekday_mean', 'hour_cos', 'diff_15min', 'same_weekday_std',
                                 'error_mlr', 'diff_90min']
site7_features_svr_reforecast = ['same_time_mean', 'Total solar irradiance (W/m2)', 'hour_sin', 'hour_cos',
                                 'diff_15min', 'lag_8', 'same_weekday_mean', 'error_mlr', 'same_time_max']
site8_features_svr_reforecast = ['hour_sin', 'same_time_mean', 'diff_15min', 'hour_cos', 'diff_30min', 'same_time_max',
                                 'diff_90min', 'diff_60min', 'same_weekday_mean', 'diff_45min', 'diff_75min']

site1_features_lgb_reforecast = ['error_mlr', 'hour_cos', 'hour_sin', 'Power (MW)', 'Total solar irradiance (W/m2)',
                                 'lag_1']
site2_features_lgb_reforecast = ['error_mlr', 'hour_cos', 'hour_sin', 'diff_105min', 'diff_45min']
site4_features_lgb_reforecast = ['hour_sin', 'error_mlr', 'diff_105min', 'lag_8', 'diff_15min', 'diff_90min', 'lag_1',
                                 'diff_60min']
site5_features_lgb_reforecast = ['hour_sin', 'error_mlr', 'lag_8', 'diff_15min']
site6_features_lgb_reforecast = ['error_mlr', 'hour_sin', 'hour_cos', 'Total solar irradiance (W/m2)', 'lag_8',
                                 'diff_105min', 'diff_90min', 'lag_1', 'diff_60min']
site7_features_lgb_reforecast = ['error_mlr', 'hour_sin', 'hour_cos', 'lag_8', 'lag_1', 'Power (MW)']
site8_features_lgb_reforecast = ['hour_sin', 'diff_105min', 'error_mlr', 'diff_60min']

site1_features_mlp_reforecast = ['same_time_max', 'Total solar irradiance (W/m2)', 'lag_7', 'Power (MW)', 'lag_2']
site2_features_mlp_reforecast = ['hour_cos', 'hour_sin', 'lag_6', 'Total solar irradiance (W/m2)', 'lag_5',
                                 'diff_60min']
site4_features_mlp_reforecast = ['hour_sin', 'same_weekday_std', 'Total solar irradiance (W/m2)', 'same_weekday_mean']
site5_features_mlp_reforecast = ['lag_8', 'hour_sin', 'Total solar irradiance (W/m2)', 'same_time_max', 'hour_cos',
                                 'lag_6']
site6_features_mlp_reforecast = ['same_time_max', 'same_time_mean', 'Total solar irradiance (W/m2)', 'hour_sin',
                                 'hour_cos', 'lag_5', 'lag_8', 'lag_7', 'diff_90min']
site7_features_mlp_reforecast = ['hour_cos', 'hour_sin', 'Total solar irradiance (W/m2)', 'same_time_max',
                                 'same_weekday_std', 'lag_7']
site8_features_mlp_reforecast = ['same_time_max', 'hour_sin', 'same_time_mean', 'same_weekday_mean', 'same_weekday_std',
                                 'lag_8', 'diff_105min', 'Power (MW)', 'diff_90min', 'lag_7', 'hour_cos']

site1_features_list = [site1_features_mlr_reforecast, site1_features_svr_reforecast, site1_features_lgb_reforecast,
                       site1_features_mlp_reforecast]
site2_features_list = [site2_features_mlr_reforecast, site2_features_svr_reforecast, site2_features_lgb_reforecast,
                       site2_features_mlp_reforecast]
site4_features_list = [site4_features_mlr_reforecast, site4_features_svr_reforecast, site4_features_lgb_reforecast,
                       site4_features_mlp_reforecast]
site5_features_list = [site5_features_mlr_reforecast, site5_features_svr_reforecast, site5_features_lgb_reforecast,
                       site5_features_mlp_reforecast]
site6_features_list = [site6_features_mlr_reforecast, site6_features_svr_reforecast, site6_features_lgb_reforecast,
                       site6_features_mlp_reforecast]
site7_features_list = [site7_features_mlr_reforecast, site7_features_svr_reforecast, site7_features_lgb_reforecast,
                       site7_features_mlp_reforecast]
site8_features_list = [site8_features_mlr_reforecast, site8_features_svr_reforecast, site8_features_lgb_reforecast,
                       site8_features_mlp_reforecast]

# MLR forecast 기준 재예측 피처 묶음 (사이트별)
site_features_mlr_forecast = [site1_features_list, site2_features_list, site4_features_list, site5_features_list,
                              site6_features_list, site7_features_list, site8_features_list]

# =====================================================================================
# 2) SVR forecast에 대한 reforecast feature 리스트 (사이트별)
#    - 아래에서 변수명이 다시 siteX_features_mlr_reforecast 등으로 재사용되지만,
#      Python 상에서는 "다음 블록에서 덮어쓰는 값"이라 논리적으로는 별도의 용도로 쓰임.
# =====================================================================================

# svr forecast reforecast features
site1_features_mlr_reforecast = ['same_time_max', 'lag_8', 'Total solar irradiance (W/m2)', 'same_time_mean',
                                 'same_time_min', 'diff_105min', 'same_time_std']
site2_features_mlr_reforecast = ['same_time_mean', 'diff_15min', 'error_svr', 'lag_8', 'Total solar irradiance (W/m2)',
                                 'hour_cos', 'pred_svr']
site4_features_mlr_reforecast = ['lag_8', 'error_svr', 'same_time_max', 'diff_15min', 'Total solar irradiance (W/m2)',
                                 'lag_1']
site5_features_mlr_reforecast = ['Total solar irradiance (W/m2)', 'same_weekday_std', 'error_svr', 'overall_min',
                                 'lag_8', 'same_weekday_min', 'same_time_max']
site6_features_mlr_reforecast = ['Total solar irradiance (W/m2)', 'same_time_max', 'lag_8', 'lag_7']
site7_features_mlr_reforecast = ['Total solar irradiance (W/m2)', 'same_time_max', 'same_time_mean', 'hour_cos']
site8_features_mlr_reforecast = ['lag_8', 'same_time_max', 'same_weekday_mean', 'same_weekday_max', 'lag_7',
                                 'same_time_mean', 'hour_cos', 'lag_2']

site1_features_svr_reforecast = ['diff_15min', 'Total solar irradiance (W/m2)', 'error_svr', 'same_time_max', 'lag_8',
                                 'hour_sin']
site2_features_svr_reforecast = ['error_svr', 'diff_15min', 'lag_1']
site4_features_svr_reforecast = ['error_svr', 'diff_15min', 'hour_sin', 'lag_1', 'Total solar irradiance (W/m2)',
                                 'lag_8', 'Power (MW)']
site5_features_svr_reforecast = ['error_svr', 'diff_15min', 'hour_sin', 'Total solar irradiance (W/m2)', 'lag_1',
                                 'same_time_max', 'diff_30min']
site6_features_svr_reforecast = ['Total solar irradiance (W/m2)', 'error_svr', 'diff_15min']
site7_features_svr_reforecast = ['Total solar irradiance (W/m2)', 'diff_15min', 'error_svr']
site8_features_svr_reforecast = ['diff_15min', 'error_svr', 'same_time_mean', 'hour_sin']

site1_features_lgb_reforecast = ['Power (MW)', 'Total solar irradiance (W/m2)', 'lag_7', 'pred_svr', 'lag_1']
site2_features_lgb_reforecast = ['hour_cos', 'error_svr', 'Total solar irradiance (W/m2)', 'lag_1', 'hour_sin',
                                 'diff_30min', 'Power (MW)', 'diff_15min', 'pred_svr', 'diff_90min', 'diff_75min']
site4_features_lgb_reforecast = ['hour_sin', 'Power (MW)', 'error_svr', 'lag_1', 'hour_cos', 'diff_30min', 'lag_8',
                                 'lag_2', 'lag_6', 'lag_7', 'diff_15min']
site5_features_lgb_reforecast = ['hour_sin', 'Total solar irradiance (W/m2)', 'lag_2', 'Power (MW)']
site6_features_lgb_reforecast = ['Total solar irradiance (W/m2)', 'hour_cos', 'same_time_max', 'same_time_mean',
                                 'lag_1']
site7_features_lgb_reforecast = ['Total solar irradiance (W/m2)', 'same_time_max', 'hour_cos', 'error_svr', 'pred_svr',
                                 'Power (MW)']
site8_features_lgb_reforecast = ['hour_cos', 'error_svr', 'same_time_max', 'Total solar irradiance (W/m2)']

site1_features_mlp_reforecast = ['same_time_max', 'Total solar irradiance (W/m2)', 'same_time_mean', 'hour_sin',
                                 'lag_2', 'same_time_std', 'diff_90min']
site2_features_mlp_reforecast = ['hour_cos', 'hour_sin', 'lag_1', 'lag_6', 'Total solar irradiance (W/m2)',
                                 'diff_90min']
site4_features_mlp_reforecast = ['same_time_max', 'hour_sin', 'same_time_std', 'lag_3', 'Total solar irradiance (W/m2)',
                                 'same_weekday_std', 'same_weekday_mean', 'lag_8', 'lag_4', 'diff_105min', 'diff_90min',
                                 'lag_6', 'Power (MW)', 'error_svr']
site5_features_mlp_reforecast = ['hour_sin', 'same_time_mean', 'lag_3', 'same_time_max',
                                 'Total solar irradiance (W/m2)', 'same_weekday_min', 'same_time_std',
                                 'same_weekday_std', 'overall_min', 'same_weekday_mean']
site6_features_mlp_reforecast = ['Total solar irradiance (W/m2)', 'lag_1', 'hour_sin', 'pred_svr', 'lag_2']
site7_features_mlp_reforecast = ['Total solar irradiance (W/m2)', 'same_time_mean', 'lag_3', 'hour_sin']
site8_features_mlp_reforecast = ['lag_2', 'lag_8', 'same_time_mean', 'lag_4', 'Total solar irradiance (W/m2)', 'lag_1',
                                 'pred_svr', 'lag_7', 'diff_45min', 'same_weekday_max']

site1_features_list = [site1_features_mlr_reforecast, site1_features_svr_reforecast, site1_features_lgb_reforecast,
                       site1_features_mlp_reforecast]
site2_features_list = [site2_features_mlr_reforecast, site2_features_svr_reforecast, site2_features_lgb_reforecast,
                       site2_features_mlp_reforecast]
site4_features_list = [site4_features_mlr_reforecast, site4_features_svr_reforecast, site4_features_lgb_reforecast,
                       site4_features_mlp_reforecast]
site5_features_list = [site5_features_mlr_reforecast, site5_features_svr_reforecast, site5_features_lgb_reforecast,
                       site5_features_mlp_reforecast]
site6_features_list = [site6_features_mlr_reforecast, site6_features_svr_reforecast, site6_features_lgb_reforecast,
                       site6_features_mlp_reforecast]
site7_features_list = [site7_features_mlr_reforecast, site7_features_svr_reforecast, site7_features_lgb_reforecast,
                       site7_features_mlp_reforecast]
site8_features_list = [site8_features_mlr_reforecast, site8_features_svr_reforecast, site8_features_lgb_reforecast,
                       site8_features_mlp_reforecast]

# SVR forecast 기준 재예측 피처 묶음
site_features_svr_forecast = [site1_features_list, site2_features_list, site4_features_list, site5_features_list,
                              site6_features_list, site7_features_list, site8_features_list]

# =====================================================================================
# 3) LGB forecast에 대한 reforecast feature 리스트 (사이트별)
# =====================================================================================

# lgb forecast reforecast features
site1_features_mlr_reforecast = ['same_time_mean', 'same_weekday_mean', 'Air temperature  (°C)', 'same_weekday_std',
                                 'lag_8', 'overall_mean', 'diff_15min', 'overall_std', 'error_lgb', 'lag_6']
site2_features_mlr_reforecast = ['same_time_max', 'lag_8', 'diff_15min', 'same_time_std', 'error_lgb',
                                 'Total solar irradiance (W/m2)']
site4_features_mlr_reforecast = ['error_lgb', 'diff_15min', 'same_time_std', 'overall_mean']
site5_features_mlr_reforecast = ['overall_min', 'same_weekday_min', 'same_time_max', 'error_lgb', 'same_weekday_std',
                                 'Air temperature  (°C)', 'same_time_std', 'diff_15min', 'same_weekday_mean', 'lag_8',
                                 'month_cos', 'month_sin']
site6_features_mlr_reforecast = ['Total solar irradiance (W/m2)', 'diff_15min']
site7_features_mlr_reforecast = ['Total solar irradiance (W/m2)', 'lag_8']
site8_features_mlr_reforecast = ['error_lgb', 'diff_15min', 'same_time_max', 'same_weekday_std', 'diff_30min',
                                 'same_weekday_max', 'same_time_std', 'diff_75min']

site1_features_svr_reforecast = ['diff_30min', 'Total solar irradiance (W/m2)', 'hour_cos', 'diff_15min', 'error_lgb',
                                 'hour_sin']
site2_features_svr_reforecast = ['diff_15min', 'hour_cos', 'diff_30min', 'same_time_max', 'error_lgb',
                                 'Total solar irradiance (W/m2)', 'hour_sin', 'lag_2', 'same_time_mean', 'diff_45min']
site4_features_svr_reforecast = ['hour_cos', 'error_lgb', 'diff_15min', 'same_time_max', 'diff_30min', 'same_time_std',
                                 'Total solar irradiance (W/m2)', 'diff_60min']
site5_features_svr_reforecast = ['same_time_max', 'diff_15min', 'hour_cos', 'diff_30min', 'same_time_std', 'pred_lgb']
site6_features_svr_reforecast = ['Total solar irradiance (W/m2)', 'same_time_mean', 'lag_2']
site7_features_svr_reforecast = ['Total solar irradiance (W/m2)', 'hour_cos', 'same_time_mean', 'diff_15min']
site8_features_svr_reforecast = ['error_lgb', 'diff_15min', 'same_time_mean', 'same_time_max', 'hour_sin', 'diff_30min',
                                 'lag_8', 'same_time_std', 'diff_45min']

site1_features_lgb_reforecast = ['error_lgb', 'pred_lgb', 'diff_15min', 'diff_30min']
site2_features_lgb_reforecast = ['error_lgb', 'lag_5', 'diff_15min', 'same_time_min', 'pred_lgb', 'diff_105min',
                                 'Total solar irradiance (W/m2)', 'lag_1', 'diff_30min', 'diff_75min', 'same_time_max',
                                 'Atmosphere (hpa)']
site4_features_lgb_reforecast = ['error_lgb', 'diff_15min', 'Power (MW)', 'hour_sin']
site5_features_lgb_reforecast = ['error_lgb', 'diff_15min', 'pred_lgb', 'diff_60min', 'Power (MW)']
site6_features_lgb_reforecast = ['Total solar irradiance (W/m2)', 'error_lgb', 'lag_5', 'pred_lgb', 'diff_15min',
                                 'lag_4', 'lag_1', 'Power (MW)', 'diff_30min', 'lag_2']
site7_features_lgb_reforecast = ['Total solar irradiance (W/m2)', 'pred_lgb', 'error_lgb', 'diff_15min']
site8_features_lgb_reforecast = ['error_lgb', 'diff_15min', 'diff_75min']

site1_features_mlp_reforecast = ['same_time_max', 'lag_7', 'Total solar irradiance (W/m2)']
site2_features_mlp_reforecast = ['lag_7', 'lag_8', 'hour_sin', 'diff_15min']
site4_features_mlp_reforecast = ['diff_15min', 'error_lgb', 'Total solar irradiance (W/m2)', 'diff_105min', 'pred_lgb',
                                 'lag_1']
site5_features_mlp_reforecast = ['hour_cos', 'error_lgb', 'diff_15min', 'month_cos', 'lag_7', 'overall_min',
                                 'Total solar irradiance (W/m2)', 'month_sin', 'same_time_std', 'diff_105min',
                                 'same_time_mean']
site6_features_mlp_reforecast = ['Total solar irradiance (W/m2)', 'hour_cos', 'pred_lgb', 'lag_4', 'lag_2',
                                 'diff_15min', 'hour_sin']
site7_features_mlp_reforecast = ['Total solar irradiance (W/m2)', 'pred_lgb', 'lag_3', 'same_weekday_std']
site8_features_mlp_reforecast = ['same_time_std', 'same_time_max', 'diff_45min', 'error_lgb', 'diff_15min',
                                 'same_time_mean', 'hour_cos', 'diff_105min', 'lag_3', 'pred_lgb', 'lag_2']

site1_features_list = [site1_features_mlr_reforecast, site1_features_svr_reforecast, site1_features_lgb_reforecast,
                       site1_features_mlp_reforecast]
site2_features_list = [site2_features_mlr_reforecast, site2_features_svr_reforecast, site2_features_lgb_reforecast,
                       site2_features_mlp_reforecast]
site4_features_list = [site4_features_mlr_reforecast, site4_features_svr_reforecast, site4_features_lgb_reforecast,
                       site4_features_mlp_reforecast]
site5_features_list = [site5_features_mlr_reforecast, site5_features_svr_reforecast, site5_features_lgb_reforecast,
                       site5_features_mlp_reforecast]
site6_features_list = [site6_features_mlr_reforecast, site6_features_svr_reforecast, site6_features_lgb_reforecast,
                       site6_features_mlp_reforecast]
site7_features_list = [site7_features_mlr_reforecast, site7_features_svr_reforecast, site7_features_lgb_reforecast,
                       site7_features_mlp_reforecast]
site8_features_list = [site8_features_mlr_reforecast, site8_features_svr_reforecast, site8_features_lgb_reforecast,
                       site8_features_mlp_reforecast]

# LGB forecast 기준 재예측 피처 묶음
site_features_lgb_forecast = [site1_features_list, site2_features_list, site4_features_list, site5_features_list,
                              site6_features_list, site7_features_list, site8_features_list]

# =====================================================================================
# 4) MLP forecast에 대한 reforecast feature 리스트 (사이트별)
# =====================================================================================

# mlp forecast reforecast features
site1_features_mlr_reforecast = ['same_time_max', 'same_time_std', 'diff_45min', 'hour_cos',
                                 'Total solar irradiance (W/m2)', 'lag_8', 'same_weekday_mean', 'diff_105min',
                                 'hour_sin']
site2_features_mlr_reforecast = ['same_time_min', 'lag_8', 'same_time_max', 'same_time_mean',
                                 'Total solar irradiance (W/m2)', 'Air temperature  (°C)', 'diff_15min', 'lag_1']
site4_features_mlr_reforecast = ['same_time_std', 'hour_cos', 'same_time_mean', 'same_time_min', 'same_time_max',
                                 'lag_8', 'Atmosphere (hpa)', 'same_weekday_std', 'error_mlp']
site5_features_mlr_reforecast = ['same_time_std', 'same_weekday_std', 'same_weekday_mean', 'diff_15min',
                                 'same_time_max', 'same_time_min', 'same_weekday_min', 'Total solar irradiance (W/m2)',
                                 'month_cos']
site6_features_mlr_reforecast = ['same_time_max', 'same_time_mean', 'lag_8', 'Total solar irradiance (W/m2)',
                                 'same_time_min', 'error_mlp']
site7_features_mlr_reforecast = ['Total solar irradiance (W/m2)', 'same_time_max', 'lag_2', 'same_time_mean']
site8_features_mlr_reforecast = ['Total solar irradiance (W/m2)', 'error_mlp', 'same_weekday_mean', 'overall_std',
                                 'same_weekday_max']

site1_features_svr_reforecast = ['error_mlp', 'diff_15min', 'Total solar irradiance (W/m2)', 'hour_cos', 'diff_45min',
                                 'same_time_mean', 'diff_60min']
site2_features_svr_reforecast = ['error_mlp', 'diff_15min', 'hour_cos']
site4_features_svr_reforecast = ['error_mlp', 'hour_sin', 'diff_15min']
site5_features_svr_reforecast = ['error_mlp', 'diff_15min', 'hour_sin', 'pred_mlp', 'same_time_max', 'lag_1']
site6_features_svr_reforecast = ['Total solar irradiance (W/m2)', 'error_mlp', 'hour_sin']
site7_features_svr_reforecast = ['error_mlp', 'diff_15min', 'hour_cos', 'same_time_mean',
                                 'Total solar irradiance (W/m2)', 'lag_8']
site8_features_svr_reforecast = ['error_mlp', 'diff_15min', 'Total solar irradiance (W/m2)', 'same_time_mean',
                                 'hour_sin', 'same_time_std', 'diff_30min', 'diff_45min', 'same_time_min',
                                 'same_time_max']

site1_features_lgb_reforecast = ['error_mlp', 'Total solar irradiance (W/m2)', 'diff_105min', 'diff_60min',
                                 'diff_75min', 'hour_cos', 'diff_45min', 'same_time_mean', 'lag_6', 'diff_90min',
                                 'diff_30min', 'diff_15min']
site2_features_lgb_reforecast = ['error_mlp', 'diff_75min', 'same_time_min', 'pred_mlp', 'diff_90min', 'lag_4',
                                 'diff_30min', 'diff_60min']
site4_features_lgb_reforecast = ['error_mlp', 'lag_1', 'Power (MW)', 'diff_30min', 'diff_45min', 'same_time_min',
                                 'same_time_std']
site5_features_lgb_reforecast = ['error_mlp', 'same_time_std', 'Power (MW)', 'diff_90min', 'diff_30min', 'diff_15min',
                                 'diff_105min', 'diff_60min']
site6_features_lgb_reforecast = ['lag_5', 'error_mlp', 'diff_15min', 'same_time_min', 'same_time_mean', 'diff_60min']
site7_features_lgb_reforecast = ['lag_3', 'error_mlp', 'pred_mlp', 'lag_1', 'Total solar irradiance (W/m2)',
                                 'Power (MW)', 'lag_2', 'lag_4', 'lag_5', 'diff_105min']
site8_features_lgb_reforecast = ['Total solar irradiance (W/m2)', 'pred_mlp', 'hour_sin', 'Power (MW)', 'error_mlp',
                                 'Atmosphere (hpa)', 'diff_15min', 'overall_max', 'lag_7']

site1_features_mlp_reforecast = ['lag_7', 'Power (MW)', 'same_time_max', 'lag_6', 'lag_3',
                                 'Total solar irradiance (W/m2)', 'same_time_mean', 'pred_mlp', 'diff_90min',
                                 'hour_sin', 'diff_45min', 'lag_8', 'diff_105min']
site2_features_mlp_reforecast = ['hour_sin', 'same_time_mean', 'lag_8', 'lag_1', 'Power (MW)', 'lag_2', 'lag_4']
site4_features_mlp_reforecast = ['same_time_std', 'lag_7', 'Total solar irradiance (W/m2)', 'same_time_max', 'hour_cos',
                                 'diff_15min', 'error_mlp', 'lag_2', 'lag_3', 'same_weekday_mean', 'lag_8',
                                 'same_weekday_std']
site5_features_mlp_reforecast = ['Total solar irradiance (W/m2)', 'Power (MW)', 'lag_1', 'hour_cos', 'same_time_mean',
                                 'same_weekday_std']
site6_features_mlp_reforecast = ['diff_105min', 'lag_2', 'lag_1', 'Power (MW)', 'Total solar irradiance (W/m2)',
                                 'hour_sin', 'hour_cos', 'diff_60min', 'same_time_std', 'lag_5', 'same_time_mean',
                                 'same_time_max', 'diff_75min', 'lag_7']
site7_features_mlp_reforecast = ['lag_1', 'hour_cos', 'lag_2', 'lag_7', 'lag_3', 'pred_mlp',
                                 'Total solar irradiance (W/m2)', 'same_time_mean', 'diff_105min', 'hour_sin',
                                 'same_time_max', 'same_time_min']
site8_features_mlp_reforecast = ['same_time_mean', 'diff_45min', 'lag_3', 'hour_sin', 'Total solar irradiance (W/m2)',
                                 'lag_1', 'same_time_max', 'lag_7', 'diff_15min', 'error_mlp', 'pred_mlp',
                                 'same_time_std', 'same_weekday_std', 'diff_105min', 'hour_cos', 'same_weekday_mean',
                                 'lag_8']

site1_features_list = [site1_features_mlr_reforecast, site1_features_svr_reforecast, site1_features_lgb_reforecast,
                       site1_features_mlp_reforecast]
site2_features_list = [site2_features_mlr_reforecast, site2_features_svr_reforecast, site2_features_lgb_reforecast,
                       site2_features_mlp_reforecast]
site4_features_list = [site4_features_mlr_reforecast, site4_features_svr_reforecast, site4_features_lgb_reforecast,
                       site4_features_mlp_reforecast]
site5_features_list = [site5_features_mlr_reforecast, site5_features_svr_reforecast, site5_features_lgb_reforecast,
                       site5_features_mlp_reforecast]
site6_features_list = [site6_features_mlr_reforecast, site6_features_svr_reforecast, site6_features_lgb_reforecast,
                       site6_features_mlp_reforecast]
site7_features_list = [site7_features_mlr_reforecast, site7_features_svr_reforecast, site7_features_lgb_reforecast,
                       site7_features_mlp_reforecast]
site8_features_list = [site8_features_mlr_reforecast, site8_features_svr_reforecast, site8_features_lgb_reforecast,
                       site8_features_mlp_reforecast]

# MLP forecast 기준 재예측 피처 묶음
site_features_mlp_forecast = [site1_features_list, site2_features_list, site4_features_list, site5_features_list,
                              site6_features_list, site7_features_list, site8_features_list]

# forecast 모델별로 [MLR forecast 기준, SVR forecast 기준, LGB forecast 기준, MLP forecast 기준] 피처 모음
feature_lists = [site_features_mlr_forecast, site_features_svr_forecast, site_features_lgb_forecast,
                 site_features_mlp_forecast]


def get_files_in_directory(path: str):
    """
    reforecast 단계에 사용할 CSV 파일들을 디렉터리에서 로드하여
    forecast 모델별(mlr, svr, lgb, mlp) DataFrame 리스트로 반환하는 함수.

    현재 구현:
        - mlr_files: path/result_of_paper/reforecast/ 에서 'mlr' 포함 CSV
        - svr_files: listdir는 reforecast 디렉터리를 보지만, join은 ann 디렉터리와 결합
        - lgb_files, mlp_files도 유사 패턴

    주의:
        - ann / reforecast 디렉터리 구조가 실제와 다르면 파일을 찾지 못할 수 있음.
        - 파일 이름 패턴(mlr, svr, lgb, mlp 포함)에 따라 로딩 대상이 결정됨.

    Args:
        path (str): 상위 경로 (예: './dataset').

    Returns:
        list: [mlr_df_list, svr_df_list, lgb_df_list, mlp_df_list] 형태의 리스트.
    """
    mlr_files = sorted(
        [os.path.join(path + '/result_of_paper/reforecast/', file) for file in
         os.listdir(path + '/result_of_paper/reforecast/') if
         file.endswith('.csv') and 'mlr' in file])
    svr_files = sorted(
        [os.path.join(path + '/result_of_paper/ann/', file) for file in
         os.listdir(path + '/result_of_paper/reforecast/') if
         file.endswith('.csv') and 'svr' in file])
    lgb_files = sorted(
        [os.path.join(path + '/result_of_paper/ann/', file) for file in
         os.listdir(path + '/result_of_paper/reforecast/') if
         file.endswith('.csv') and 'lgb' in file])
    mlp_files = sorted(
        [os.path.join(path + '/result_of_paper/ann/', file) for file in
         os.listdir(path + '/result_of_paper/reforecast/') if
         file.endswith('.csv') and 'mlp' in file])

    mlr_df_list = []
    svr_df_list = []
    lgb_df_list = []
    mlp_df_list = []

    for file in mlr_files:
        mlr_df_list.append(pd.read_csv(file))
    for file in svr_files:
        svr_df_list.append(pd.read_csv(file))
    for file in lgb_files:
        lgb_df_list.append(pd.read_csv(file))
    for file in mlp_files:
        mlp_df_list.append(pd.read_csv(file))

    return [mlr_df_list, svr_df_list, lgb_df_list, mlp_df_list]


def do_reforecast(origin_df_lists, forecast_models, targets, shifted_targets, site_names, test_size, path):
    """
    forecast 모델별(MLR/SVR/LGB/MLP)에 대해:
        - 재예측용 피처(feature_lists)를 사용하여
        - 오차 타깃(error_*)의 t+1 값을 재예측(pred_error_*)하고
        - 재예측된 PV(reforecasted_PV_*)를 계산·저장하는 함수.

    처리 흐름:
        1. origin_df_lists: 각 forecast 모델(mlr, svr, lgb, mlp)에 대한 DataFrame 리스트 모음.
        2. feature_lists: 각 forecast 모델 및 사이트별 재예측 피처 집합.
        3. 각 (origin_df_list, forecast_model, target, shifted_target, final_result_list, feature_list)에 대해:
           - shifted_target = target.shift(-1) 설정 후 타깃 시계열 생성.
           - 시계열 순서 기반 train/test 분할.
           - MLR/SVR/LGB/MLP로 오차를 재예측(pred_error_*).
           - 한 스텝 shift하여 t+1 시점에 대응.
           - origin_df에서 기존 예측값(pred_{forecast_model})을 merge.
           - reforecasted_PV_{model} = max(pred_{forecast_model} - pred_error_{model}, 0).
           - CSV로 저장 및 final_result_list에 추가.

    Args:
        origin_df_lists (list): [mlr_df_list, svr_df_list, lgb_df_list, mlp_df_list] 구조.
        forecast_models (list): ['mlr', 'svr', 'lgb', 'mlp'] 등 forecast 모델 이름 리스트.
        targets (list): ['error_mlr', 'error_svr', 'error_lgb', 'error_mlp'] 등 오차 타깃 컬럼명 리스트.
        shifted_targets (list): shift된 오차 타깃 컬럼명 리스트.
        site_names (list): 사이트 ID 리스트.
        test_size (float): 테스트 비율.
        path (str): 모델 및 결과 저장 경로의 상위 디렉터리.

    Returns:
        list: forecast 모델별 final_result_list를 담은 리스트
              (예: [final_list_for_mlr, final_list_for_svr, ...]).
    """
    final_result_lists = [[], [], [], []]
    for origin_df_list, forecast_model, target, shifted_target, final_result_list, final_result_train_list, feature_list in zip(
            origin_df_lists, forecast_models, targets, shifted_targets, final_result_lists, feature_lists):

        print('============================================================================================')
        print(f'forecast model : {forecast_model.upper()}')

        reforecasted_df_list = []
        for df, df_index, features in zip(origin_df_list, site_names, feature_list):
            print('----------------------------------------------------------------------------------------------')
            print(df_index)

            # 오차 타깃을 t+1 시점으로 shift
            df[shifted_target] = df[target].shift(-1)
            df = df.dropna().reset_index(drop=True)

            X_total = df.copy()
            y = df[shifted_target]
            time_col = df['Time']

            # 시계열 순서 기반 분할
            split_index = int(len(X_total) * (1 - test_size))

            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            time_train, time_test = time_col.iloc[:split_index], time_col.iloc[split_index:]

            # 정규화 스케일러
            scaler = StandardScaler()

            # ---------------- MLR ----------------
            print("MLR")

            filtered_features = [col for col in features[0] if col in df.columns]
            X = df[filtered_features]
            X_train_mlr, X_test_mlr = X.iloc[:split_index], X.iloc[split_index:]
            X_train_scaled_mlr = scaler.fit_transform(X_train_mlr)
            X_test_scaled_mlr = scaler.transform(X_test_mlr)

            mlr = LinearRegression()
            mlr.fit(X_train_scaled_mlr, y_train)

            # ---------------- SVR ----------------
            print("SVR")

            filtered_features = [col for col in features[1] if col in df.columns]
            X = df[filtered_features]
            X_train_svr, X_test_svr = X.iloc[:split_index], X.iloc[split_index:]
            X_train_scaled_svr = scaler.fit_transform(X_train_svr)
            X_test_scaled_svr = scaler.transform(X_test_svr)

            svr = SVR()
            svr.fit(X_train_scaled_svr, y_train)

            # ---------------- LGB ----------------
            print("LightGBM")

            filtered_features = [col for col in features[2] if col in df.columns]
            X = df[filtered_features]
            X_train_lgb, X_test_lgb = X.iloc[:split_index], X.iloc[split_index:]

            lgb_model = lgb.LGBMRegressor()
            lgb_model.fit(X_train_lgb, y_train)

            # ---------------- MLP ----------------
            print("MLP")

            filtered_features = [col for col in features[3] if col in df.columns]
            X = df[filtered_features]
            X_train_mlp, X_test_mlp = X.iloc[:split_index], X.iloc[split_index:]
            X_train_scaled_mlp = scaler.fit_transform(X_train_mlp)
            X_test_scaled_mlp = scaler.transform(X_test_mlp)

            mlp = MLPRegressor(random_state=42, max_iter=1000)
            mlp.fit(X_train_scaled_mlp, y_train)

            # 예측 결과를 t+1 위치에 저장 (오차 재예측)
            df_result = X_total.copy()
            df_result.loc[:split_index - 1, 'pred_error_mlr'] = mlr.predict(X_train_scaled_mlr)
            df_result.loc[split_index:, 'pred_error_mlr'] = mlr.predict(X_test_scaled_mlr)
            df_result.loc[:split_index - 1, 'pred_error_svr'] = svr.predict(X_train_scaled_svr)
            df_result.loc[split_index:, 'pred_error_svr'] = svr.predict(X_test_scaled_svr)
            df_result.loc[:split_index - 1, 'pred_error_lgb'] = lgb_model.predict(X_train_lgb)
            df_result.loc[split_index:, 'pred_error_lgb'] = lgb_model.predict(X_test_lgb)
            df_result.loc[:split_index - 1, 'pred_error_mlp'] = mlp.predict(X_train_scaled_mlp)
            df_result.loc[split_index:, 'pred_error_mlp'] = mlp.predict(X_test_scaled_mlp)

            # t+1 시점에 맞게 한 스텝 shift
            df_result['pred_error_mlr'] = df_result['pred_error_mlr'].shift(1)
            df_result['pred_error_svr'] = df_result['pred_error_svr'].shift(1)
            df_result['pred_error_lgb'] = df_result['pred_error_lgb'].shift(1)
            df_result['pred_error_mlp'] = df_result['pred_error_mlp'].shift(1)

            # 결과 추출 및 Time 복원
            X_test_result = df_result.copy()
            X_test_result.loc[:split_index - 1, 'Time'] = time_train.values
            X_test_result.loc[split_index:, 'Time'] = time_test.values
            X_test_result.dropna(subset=['pred_error_mlr', 'pred_error_svr', 'pred_error_lgb', 'pred_error_mlp'],
                                 inplace=True)

            reforecasted_df_list.append(X_test_result)

            # 재예측 모델 저장
            joblib.dump(mlr,
                        f"{path}/result_of_paper/feature_selection/reforecast/mlr_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            joblib.dump(svr,
                        f"{path}/result_of_paper/feature_selection/reforecast/svr_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")
            lgb_model.booster_.save_model(
                f"{path}/result_of_paper/feature_selection/reforecast/lgb_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.txt")
            joblib.dump(mlp,
                        f"{path}/result_of_paper/feature_selection/reforecast/mlp_reforecasting_model_{forecast_model}_forecast_{str(df_index)}.joblib")

        # 원래 forecast 결과(origin_df)와 병합하여 실제 재예측된 PV 계산
        for origin_df, forecasted_df in zip(origin_df_list, reforecasted_df_list):
            # 기존 pred_{forecast_model}가 있다면 일단 제거 후 origin_df 기준으로 merge
            forecasted_df = forecasted_df[[col for col in forecasted_df.columns if col != f'pred_{forecast_model}']]
            forecasted_df = forecasted_df.merge(
                origin_df[['Time', f'pred_{forecast_model}']],
                on='Time',
                how='left'
            )

            # 재예측된 PV 계산 (예측값 - 예측된 오차, 음수는 0으로 클리핑)
            forecasted_df['reforecasted_PV_mlr'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_mlr']), 0)
            forecasted_df['reforecasted_PV_svr'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_svr']), 0)
            forecasted_df['reforecasted_PV_lgb'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_lgb']), 0)
            forecasted_df['reforecasted_PV_mlp'] = np.maximum(
                (forecasted_df[f'pred_{forecast_model}'] - forecasted_df['pred_error_mlp']), 0)

            final_result_list.append(forecasted_df)
            forecasted_df.to_csv(
                f"{path}/result_of_paper/feature_selection/reforecast/{forecast_model}_forecast_reforecasted_{str(df_index)}.csv",
                index=False)

    return final_result_lists


def do_evaluate(final_result_lists, site_names, path, test_size, operation_hours_array, reforecast_models, target,
                forecast_models):
    """
    재예측 결과(final_result_lists)에 대해,
    운전 시간대별 성능 지표(MAE, MSE, RMSE)를 출력하는 함수.

    Args:
        final_result_lists (list): forecast 모델별 재예측 결과 DataFrame 리스트 모음.
        site_names (list): 사이트 ID 리스트.
        path (str): (현재 함수에서는 사용하지 않지만) 상위 경로 인자로 전달됨.
        test_size (float): train/test 분할 시 사용한 테스트 비율.
        operation_hours_array (list): 각 사이트별 (시작시각, 종료시각) 튜플 리스트.
        reforecast_models (list): ['mlr', 'svr', 'lgb', 'mlp'] 등 재예측 모델 이름 리스트.
        target (str): 실제 타깃 컬럼명 (예: 'Power (MW)').
        forecast_models (list): ['mlr', 'svr', 'lgb', 'mlp'] 등 forecast 모델 이름 리스트.

    Returns:
        None
    """
    for forecast_model, final_result_list in zip(forecast_models, final_result_lists):
        print('==========================================================')
        print(f'forecast model : {forecast_model.upper()}')

        for df_index, (start_str, end_str), final_result in zip(site_names, operation_hours_array, final_result_list):
            print('----------------------------------------------------------')
            print(df_index)
            for reforecast_model in reforecast_models:
                print(f'reforecast model : {reforecast_model.upper()}')

                df = final_result.copy()
                split_index = int(len(df) * (1 - test_size))

                df_copy = df.iloc[split_index:].copy()
                df_copy['Time'] = pd.to_datetime(df_copy['Time'])

                start_time = time.fromisoformat(start_str)
                end_time = time.fromisoformat(end_str)

                result_df = df_copy[df_copy['Time'].dt.time.between(start_time, end_time)]

                mse = ((result_df[target] - result_df[f'reforecasted_PV_{reforecast_model}']) ** 2).mean()
                mae = (result_df[target] - result_df[f'reforecasted_PV_{reforecast_model}']).abs().mean()
                rmse = np.sqrt(mse)

                print(f"MAE: {mae}")
                print(f"MSE: {mse}")
                print(f"RMSE: {rmse}")


def do_reforecast_and_evaluate_and_save(path, forecast_models, reforecast_targets, reforecast_shifted_targets,
                                        site_names, test_size, operation_hours_array, reforecast_models, target):
    """
    전체 reforecast 파이프라인을 한 번에 수행하는 헬퍼 함수.

    처리 흐름:
        1. get_files_in_directory(path) 호출 → forecast 모델별 origin_df_lists 로드.
        2. do_reforecast(...) 호출 → 오차 재예측 및 재예측된 PV 계산, CSV 저장.
        3. do_evaluate(...) 호출 → 운전 시간대별 MAE/MSE/RMSE 출력.

    Args:
        path (str): 상위 경로 (예: './dataset').
        forecast_models (list): ['mlr', 'svr', 'lgb', 'mlp'] 등 forecast 모델 리스트.
        reforecast_targets (list): ['error_mlr', 'error_svr', ...] 등 재예측 타깃 컬럼명 리스트.
        reforecast_shifted_targets (list): shift된 재예측 타깃 컬럼명 리스트.
        site_names (list): 사이트 ID 리스트.
        test_size (float): 테스트 데이터 비율.
        operation_hours_array (list): 각 사이트별 운전 시간대 (시작, 종료) 튜플 리스트.
        reforecast_models (list): 재예측 모델 이름 리스트 (예: ['mlr', 'svr', 'lgb', 'mlp']).
        target (str): 실제 타깃 컬럼명 (예: 'Power (MW)').

    Returns:
        None
    """
    origin_df_lists = get_files_in_directory(path)
    final_result_lists = do_reforecast(origin_df_lists, forecast_models, reforecast_targets, reforecast_shifted_targets,
                                       site_names,test_size, path)
    do_evaluate(final_result_lists, site_names, path, test_size, operation_hours_array, reforecast_models, target,
                forecast_models)
