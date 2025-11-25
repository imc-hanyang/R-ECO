"""
PV 발전량 예측 및 재예측 전체 파이프라인을 실행하는 메인 스크립트.

- 원시 데이터를 읽어 전처리 및 피처 엔지니어링을 수행
- 기본 예측(Base forecast) 실행 및 평가
- 재예측(Reforecast) 실행 및 평가
"""

from forecast_reforecast import get_feature_added_dataframe, do_forecast_and_evaluate, \
    create_reforecast_features_and_targets, do_reforecast_and_evaluate
from handle_data_with_preprocessing import save_feature_add_df_list, get_raw_df, delete_nulls_and_add_time_column
from feature_lists import forecast_features

# 전처리 및 결과 저장에 사용할 기본 경로 설정
path = './dataset'
datasets_path = path + '/dataset/solar_stations'
# 테스트 데이터 비율 (여기서는 2개월 / 24개월)
test_size = 2 / 24

# 사용할 사이트(발전소) ID 리스트 (예: 1,2,4,5,6,7,8번 사이트)
site_names = [1, 2, 4, 5, 6, 7, 8]  # set selected df's site name e.g. ['first']

# 각 사이트별 운전 시간대(Operation hours) 설정 (시작시각, 종료시각)
operation_hours_array = [
    ('06:00', '21:30'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('06:00', '21:00'),
    ('06:00', '21:00'),
    ('06:00', '19:00')
]


def validate_lengths(site_names, operation_hours_array, feature_added_df_list):
    """
    사이트 관련 리스트들의 길이 일관성을 검증하는 함수.

    Args:
        site_names (list): 사이트 이름 또는 ID 리스트.
        operation_hours_array (list): 각 사이트별 운전 시간대 튜플 리스트.
        feature_added_df_list (list): 각 사이트별 전처리 완료 DataFrame 리스트.

    Raises:
        ValueError: 세 리스트의 길이가 서로 다를 경우 발생.
    """
    if len(site_names) != len(operation_hours_array):
        raise ValueError("site_names와 operation_hours_array의 길이는 같아야 합니다.")
    if len(site_names) != len(feature_added_df_list):
        raise ValueError("site_names와 feature_added_df_list의 길이는 같아야 합니다.")
    if len(operation_hours_array) != len(feature_added_df_list):
        raise ValueError("operation_hours_array와 feature_added_df_list의 길이는 같아야 합니다.")


def main():
    """
    전체 예측/재예측 파이프라인을 실행하는 메인 함수.

    처리 단계:
        1. 원시 데이터 로드 및 전처리 (결측치 처리 + 피처 추가).
        2. 전처리 결과를 파일로 저장.
        3. 사이트/운전시간/데이터 리스트 길이 검증.
        4. 피처가 추가된 CSV를 다시 로드.
        5. 기본 예측(Base forecast) 수행 및 결과/평가 출력.
        6. 재예측(Reforecast)에 필요한 피처/타깃 정보 생성.
        7. 재예측 수행 및 평가 출력.
    """
    df_list = get_raw_df(datasets_path)
    feature_added_df_list = delete_nulls_and_add_time_column(df_list)
    save_feature_add_df_list(feature_added_df_list)
    print(f" 첫 번째 사이트 DataFrame의 컬럼 목록 : {feature_added_df_list[0].columns()}")

    # 사이트 이름, 운전 시간대, 전처리된 DataFrame 리스트의 길이 일관성 검증
    # 전처리된 데이터(lag/feature 추가된 CSV)를 다시 로드해서 사용
    # 기본 예측(Base forecast) 수행 및 평가
    validate_lengths(site_names, operation_hours_array, feature_added_df_list)
    feature_added_df_list = get_feature_added_dataframe(path)
    forecasted_df_list = do_forecast_and_evaluate(path, forecast_features, "Power (MW)", 'power_shifted', test_size,
                                                  feature_added_df_list, site_names, operation_hours_array)

    # 재예측(Reforecast)에 필요한 피처/타깃/시프트 타깃 정보 생성
    # 재예측 수행 및 평가
    reforecast_features_list, reforecast_targets, reforecast_shifted_targets = create_reforecast_features_and_targets()
    do_reforecast_and_evaluate(reforecast_targets, reforecast_shifted_targets, reforecast_features_list, site_names,
                               forecasted_df_list, operation_hours_array, test_size, path)


if __name__ == "__main__":
    main()
