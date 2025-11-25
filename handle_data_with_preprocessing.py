"""
태양광 발전소 시계열 데이터를 전처리하고 피처를 생성하는 모듈.

- 원시 엑셀 파일을 읽어서 pandas DataFrame 리스트로 변환한다.
- 결측치를 확인하고, 보간(interpolate)을 통해 결측치를 처리한다.
- 시간 정보(월, 요일, 시각)를 기반으로 사인/코사인 주기 특성을 추가한다.
- 최근 1주일 창(window)에 대한 통계 특성(평균, 표준편차, 최소/최대 등)을 추가한다.
- 발전량 시차(lag) 및 시간 차이(diff) 특성을 생성한다.
- 최종적으로 특성이 추가된 DataFrame들을 CSV 파일로 저장한다.
"""

import os
import math
from datetime import timedelta

import numpy as np
import pandas as pd


def get_raw_df(datasets_path: str) -> (list, list):
    """
    주어진 경로에서 원시 엑셀 데이터 파일들을 읽어오는 함수.

    Args:
        datasets_path (str): 발전소 시계열 엑셀 파일이 저장된 디렉터리 경로.

    Returns:
        tuple[list, list]:
            - files: 읽어들인 엑셀 파일의 전체 경로 문자열 리스트.
            - df_list: 각 파일을 읽어 만든 pandas DataFrame 객체 리스트.
    """
    files = [
        os.path.join(datasets_path, file)
        for file in os.listdir(datasets_path)
        if file.endswith(".xlsx") and "_3_" not in file
    ]
    files.sort()

    df_list = []
    for file in files:
        print(file)
        df = pd.read_excel(file)
        df_list.append(df)

    return files, df_list


def delete_nulls_and_add_time_column(df_list: list, files: list) -> list:
    """
    결측치를 확인하고, 일부 결측치를 보간해서 제거한 뒤
    시간 관련 피처와 추가 피처들을 생성하는 함수.

    처리 단계:
        1. 각 DataFrame의 결측치 개수를 출력(check_nulls).
        2. 특정 DataFrame의 결측치를 선형 보간(delete_nulls).
        3. 시간/주간/lag 피처를 차례대로 추가(add_columns).

    Args:
        df_list (list): 원시 데이터를 담고 있는 DataFrame 리스트.
        files (list): 각 DataFrame에 대응되는 원본 파일 경로 리스트.

    Returns:
        list: 시간/주간/lag 피처가 모두 추가된 DataFrame 리스트.
    """
    check_nulls(df_list, files)
    delete_nulls(df_list)
    return add_columns(df_list)


def check_nulls(df_list: list, files: list) -> None:
    """
    각 DataFrame에 존재하는 결측치 개수를 출력하는 함수.

    Args:
        df_list (list): 결측치를 확인할 DataFrame 리스트.
        files (list): 각 DataFrame에 대응되는 원본 파일 경로 리스트.

    Returns:
        None
    """
    for i in range(0, len(df_list)):
        print(files[i])
        # isnull과 isna는 같은 의미이며, 두 가지 형태를 모두 확인하기 위해 출력
        print(df_list[i].isnull().sum())
        print(df_list[i].isna().sum())


def delete_nulls(df_list: list) -> None:
    """
    DataFrame 리스트 중 특정 인덱스(예: 2번)의 결측치를 선형 보간으로 처리하는 함수.

    현재 구현:
        - df_list[2]에서 결측치를 포함하는 행을 먼저 출력.
        - 선형 보간(interpolate)으로 결측치를 채움.
        - 보간 후 남아 있는 결측치 개수를 출력.

    Args:
        df_list (list): 결측치 처리를 적용할 DataFrame 리스트.

    Returns:
        None
    """
    # 결측치를 포함하는 행 출력
    print(df_list[2][df_list[2].isna().any(axis=1)])

    # 선형 보간으로 결측치 채우기 (원본 DataFrame을 직접 수정)
    df_list[2].interpolate(method="linear", inplace=True)

    # 보간 후 결측치 개수 확인
    print(df_list[2].isna().sum())


def add_columns(df_list: list) -> list:
    """
    시간 관련 피처, 최근 1주일 통계 피처, lag 피처를 순차적으로 추가하는 함수.

    처리 단계:
        1. add_time_columns: Time 컬럼과 월/요일/시각 및 주기형(sin/cos) 피처 추가.
        2. add_week_time_columns: 최근 7일(1주일) 창을 기준으로 통계 피처 추가.
        3. add_lag_columns: Power (MW)에 대한 시차(lag) 및 diff 피처 추가.

    Args:
        df_list (list): 원시 또는 전처리된 DataFrame 리스트.

    Returns:
        list: 모든 시간/통계/lag 피처가 추가된 DataFrame 리스트.
    """
    add_time_columns(df_list)
    feature_added_df_list = add_week_time_columns(df_list)
    return add_lag_columns(feature_added_df_list)


def add_time_columns(df_list: list) -> None:
    """
    각 DataFrame에 시간 관련 피처를 추가하는 함수.

    추가되는 내용:
        - 'Time(year-month-day h:m:s)' 컬럼을 datetime 타입의 'Time' 컬럼으로 변환.
        - month, day(요일), hour 숫자 피처 추가.
        - day/hour/month에 대해 주기형(sin, cos) 인코딩 피처 추가.
        - Time 기준으로 정렬.

    Args:
        df_list (list): 시간 피처를 추가할 DataFrame 리스트.

    Returns:
        None
    """
    for df in df_list:
        # 문자열 Time 컬럼을 datetime 타입으로 변환
        df["Time"] = pd.to_datetime(df["Time(year-month-day h:m:s)"])

        # 월, 요일(0=월요일), 시간 추출
        df["month"] = df["Time"].dt.month
        df["day"] = df["Time"].dt.dayofweek
        df["hour"] = df["Time"].dt.hour

        # 요일/시간/월에 대한 주기형 피처(sin, cos) 생성
        df["day_sin"] = np.sin(df["day"] * 2 * math.pi / 7)
        df["day_cos"] = np.cos(df["day"] * 2 * math.pi / 7)
        df["hour_sin"] = np.sin(df["hour"] * 2 * math.pi / 24)
        df["hour_cos"] = np.cos(df["hour"] * 2 * math.pi / 24)
        df["month_sin"] = np.sin(df["month"] * 2 * math.pi / 12)
        df["month_cos"] = np.cos(df["month"] * 2 * math.pi / 12)

        # 시간 기준 정렬
        df.sort_values(by="Time", inplace=True)


def add_week_time_columns(df_list: list) -> list:
    """
    각 DataFrame에 대해 '최근 7일(1주일)' 윈도우 기반 통계 피처를 추가하는 함수.

    각 시점 t에 대해:
        - [t-7일, t) 구간의 전체 발전량 통계(overall_*).
        - 같은 시각(Time.time 동일)의 발전량 통계(same_time_*).
        - 같은 요일(weekday 동일)의 발전량 통계(same_weekday_*).

    마지막에는 '2019-01-08' 이후의 데이터만 남기도록 필터링한다.

    Args:
        df_list (list): 시간 컬럼('Time')과 발전량('Power (MW)')을 포함한 DataFrame 리스트.

    Returns:
        list: 1주일 통계 피처가 추가되고, 날짜 필터링까지 적용된 DataFrame 리스트.
    """
    feature_added_df_list = []

    for df in df_list:
        df["Time"] = pd.to_datetime(df["Time"])

        def calc_stats(row):
            """
            개별 행(row)에 대해 최근 7일 윈도우의 통계량을 계산하는 내부 함수.

            Args:
                row (Series): 현재 시점의 행 데이터.

            Returns:
                Series: same_time_*, same_weekday_*, overall_* 통계량이 담긴 Series.
            """
            end = row["Time"]
            start = end - timedelta(days=7)
            window = df[(df["Time"] >= start) & (df["Time"] < end)]

            # 전체 발전량
            pow_all = window["Power (MW)"]
            # 같은 시각(시/분/초 동일)의 발전량
            pow_same_time = window[window["Time"].dt.time == end.time()]["Power (MW)"]
            # 같은 요일(weekday 동일)의 발전량
            pow_same_weekday = window[window["Time"].dt.weekday == end.weekday()][
                "Power (MW)"
            ]

            return pd.Series(
                {
                    "same_time_mean": pow_same_time.mean(),
                    "same_time_std": pow_same_time.std(),
                    "same_time_max": pow_same_time.max(),
                    "same_time_min": pow_same_time.min(),
                    "same_weekday_mean": pow_same_weekday.mean(),
                    "same_weekday_std": pow_same_weekday.std(),
                    "same_weekday_max": pow_same_weekday.max(),
                    "same_weekday_min": pow_same_weekday.min(),
                    "overall_mean": pow_all.mean(),
                    "overall_std": pow_all.std(),
                    "overall_max": pow_all.max(),
                    "overall_min": pow_all.min(),
                }
            )

        # 각 행에 대해 1주일 윈도우 통계 계산
        stats = df.apply(calc_stats, axis=1)

        # 계산된 통계 컬럼을 원본 DataFrame에 추가
        for col in stats.columns:
            df[col] = stats[col]

        feature_added_df_list.append(df)

    # 2019-01-08 이후 데이터만 남기도록 필터링
    filtered_feature_added_df_list = [
        feature_added_df[feature_added_df["Time"] >= "2019-01-08"]
        for feature_added_df in feature_added_df_list
    ]
    return filtered_feature_added_df_list


def add_lag_columns(feature_added_df_list: list, target_column: str) -> list:
    """
    발전량(Power (MW))에 대한 시차(lag) 및 차분(diff) 피처를 추가하는 함수.

    추가되는 내용:
        - lag_1 ~ lag_8: 15분 단위 시차 1~8 스텝 (예: lag_1은 15분 전 값).
        - diff_15min ~ diff_105min: 현재 값 - 과거 lag 값의 차이.
        - 결측값이 포함된 행은 dropna()로 제거.

    Args:
        feature_added_df_list (list): 1주일 통계 피처까지 추가된 DataFrame 리스트.
        target_column (str): 시차 및 차분 피처를 생성할 대상 컬럼명 (예: "Power (MW)").

    Returns:
        list: lag 및 diff 피처가 추가된 최종 DataFrame 리스트.
    """
    lag_added_df_list = []
    for i in range(len(feature_added_df_list)):
        feature_added_df = feature_added_df_list[i]
        print("-------------------------------------------------------------")

        # 15분 간격 x 1~8 step 시차 피처 생성
        for lag in [1, 2, 3, 4, 5, 6, 7, 8]:
            feature_added_df[f"lag_{lag}"] = feature_added_df[target_column].shift(lag)

        # 시간 차이(diff) 피처 생성
        feature_added_df["diff_15min"] = (
                feature_added_df[target_column] - feature_added_df[target_column].shift(1)
        )
        feature_added_df["diff_30min"] = (
                feature_added_df[target_column] - feature_added_df[target_column].shift(2)
        )
        feature_added_df["diff_45min"] = (
                feature_added_df[target_column] - feature_added_df[target_column].shift(3)
        )
        feature_added_df["diff_60min"] = (
                feature_added_df[target_column] - feature_added_df[target_column].shift(4)
        )
        feature_added_df["diff_75min"] = (
                feature_added_df[target_column] - feature_added_df[target_column].shift(5)
        )
        feature_added_df["diff_90min"] = (
                feature_added_df[target_column] - feature_added_df[target_column].shift(6)
        )
        feature_added_df["diff_105min"] = (
                feature_added_df[target_column] - feature_added_df[target_column].shift(7)
        )

        # 시차/차분 연산으로 인해 생긴 결측 행 제거
        feature_added_df.dropna(inplace=True)
        lag_added_df_list.append(feature_added_df)

    return lag_added_df_list


def save_feature_add_df_list(files: list, feature_added_df_list: list, path: str) -> None:
    """
    피처가 추가된 DataFrame 리스트를 CSV 파일로 저장하는 함수.

    저장 규칙:
        - 원본 파일 경로에서 파일명만 추출한 뒤 확장자(.csv) 제거.
        - 지정한 path 아래 'lag_added_dataset' 폴더에 같은 이름으로 저장.

    Args:
        files (list): 각 DataFrame에 대응되는 원본 파일 경로 리스트.
        feature_added_df_list (list): 최종 피처가 추가된 DataFrame 리스트.
        path (str): CSV를 저장할 상위 디렉터리 경로.

    Returns:
        None
    """
    for file, df in zip(files, feature_added_df_list):
        file_name = file.split("/")[-1].replace(".csv", "")
        print(f"saving {file_name}...")
        df.to_csv(path + "/lag_added_dataset/" + file_name, index=False)

