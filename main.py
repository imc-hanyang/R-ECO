from forecast_reforecast import get_feature_added_dataframe, do_forecast_and_evaluate, \
    create_reforecast_features_and_targets, do_reforecast_and_evaluate
from handle_data_with_preprocessing import save_feature_add_df_list, get_raw_df, delete_nulls_and_add_time_column
from feature_lists import forecast_features

path = './dataset'
datasets_path = path + '/dataset/solar_stations'
test_size = 2 / 24

site_names = [1, 2, 4, 5, 6, 7, 8]  # set selected df's site name e.g. ['first']
operation_hours_array = [
    ('06:00', '21:30'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('00:00', '23:59'),
    ('06:00', '21:00'),
    ('06:00', '21:00'),
    ('06:00', '19:00')
]

df_list = get_raw_df(datasets_path)
feature_added_df_list = delete_nulls_and_add_time_column(df_list)
save_feature_add_df_list(feature_added_df_list)
print(f" 첫 번째 사이트 DataFrame의 컬럼 목록 : {feature_added_df_list[0].columns()}")

if len(site_names) != len(operation_hours_array):
    raise ValueError("site_names와 operation_hours_array의 길이는 같아야 합니다.")
if len(site_names) != len(feature_added_df_list):
    raise ValueError("site_names와 feature_added_df_list의 길이는 같아야 합니다.")
if len(operation_hours_array) != len(feature_added_df_list):
    raise ValueError("operation_hours_array와 feature_added_df_list의 길이는 같아야 합니다.")

feature_added_df_list = get_feature_added_dataframe(path)
forecasted_df_list = do_forecast_and_evaluate(path, forecast_features, "Power (MW)", 'power_shifted', test_size,
                                              feature_added_df_list, site_names, operation_hours_array)

reforecast_features_list, reforecast_targets, reforecast_shifted_targets = create_reforecast_features_and_targets()
do_reforecast_and_evaluate(reforecast_targets, reforecast_shifted_targets, reforecast_features_list, site_names,
                           reforecast_features_list, operation_hours_array, test_size, path)

if __name__ == "__main__":
