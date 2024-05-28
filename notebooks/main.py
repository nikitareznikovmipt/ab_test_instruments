import sys
import os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from nfs_ab_tool.data_processer.data_processer import Data_Processer
from nfs_ab_tool.data_processer.pandas_processer import Pandas_Processer
from nfs_ab_tool.data_processer.sql_processer import SQL_Processer
from nfs_ab_tool.monitoring import Monitoring
from nfs_ab_tool.test_calculator.calculator import TestCalculator

import pandas as pd
from itertools import combinations
from pprint import pprint
import re

def main():
    # print(processer.calc_cum_stats_per_date(
    #     'dataset_for_monitoring',
    #     metrics={'total_utils': ['mean', 'std']},
    #     group_columns=['test_date', 'test_group']
    # ))

    df = pd.read_csv(os.path.join(os.getcwd(), 'notebooks/sample_dataset.csv'))
    df['first_date_in_test'] = pd.to_datetime(df['first_date_in_test'])
    df['test_date'] = pd.to_datetime(df['test_date'])
    binary_columns = ['city_entry_flag']
    continuous_columns = ['total_utils']

    config = {
        'engine': 'pandas',
        'metrics': {
            'continuous': continuous_columns,
            'binary': binary_columns,
        },
        'test_params': {
            "first_type_errors": [0.05],
            "second_type_errors": [0.2],
            "criterion": 'Т-тест',
            "metrics": ['city_entry_flag', 'total_utils']
        },
        'dataset_params': {
            'metric_per_time': {
                'user_id':['new_users_cnt'], 'city_entry_flag': ['sum'], 
            },
            'cumulative_metric_per_time': 
                {'user_id':['nunique'], 'city_entry_flag': ['sum',], 'total_utils': ['sum']},
        },
        'renaming_dict': {},
        'notification_params': {}
    }

    # processer = Data_Processer(engine='gp')

    # test_calc = TestCalculator(
    #     metrics=config['metrics'],
    #     test_config=config['test_params'],
    #     processer=processer
    # )

    # # test_calc._prepare_df('dataset_for_monitoring', metrics=config['test_params']['metrics'], groups=['A1', 'A2', 'B'])
    # # print(df.head())
    # num_obs = test_calc._transform_df(
    #     pd.read_csv(os.path.join(os.getcwd(), 'notebooks/test_users_cnt.csv')),
    #     metrics={
    #         'user_id': ['nunique'],
    #         },
    #     group_info={'A': 'A1', 'B': 'A2', 'renaming': {}, 'need_correction': False}
    # )
    # bin_df = test_calc._transform_df(pd.read_csv(os.path.join(os.getcwd(), 'notebooks/test_binary.csv')), metrics={'city_entry_flag': ['mean', 'std']}, group_info={'A': 'A1', 'B': 'A2', 'renaming': {}, 'need_correction': False})
    # # continuous_df = test_calc._transform_df(pd.read_csv(os.path.join(os.getcwd(), 'notebooks/test_continuous.csv')), metrics={'total_utils': ['mean', 'std']}, group_info={'A': 'A', 'B': 'B', 'renaming': {}, 'need_correction': False})
    # res = bin_df.merge(num_obs, on=['test_date', 'group A label', 'group B label'])
    # res_with_stats = test_calc._calc_test_days_per_days(res)
    # print(res_with_stats)

    monitoring = Monitoring(config, df)
    print(monitoring.prepare_metrics_dataset().head())
    # result = monitoring.prepare_test_results_dataset()
    # print(result.head())
    # result.to_excel('Результаты теста.xlsx')
    # print(result.head())
    # print(monitoring._groups_for_compare(['A1', 'A2', 'B', 'C']))


if __name__ == '__main__':
    main()