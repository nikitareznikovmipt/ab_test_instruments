import re
from itertools import combinations, product
from typing import Dict, List, Union

import pandas as pd

from nfs_ab_tool.data_processer import Data_Processer
from nfs_ab_tool.test_calculator.calculator import TestCalculator

from .config import DATE_COL, FIRST_DATE_IN_TEST, GROUP_COL, USER_COL

try:
    import tinkoff as tf  # type: ignore
except ImportError:
    tf = None


class Monitoring:
    def __init__(self, config, data_source: Union[pd.DataFrame, str]) -> None:
        self.processer = Data_Processer(config["engine"])
        self.metrics = config["metrics"]
        self.test_params = config["test_params"]
        self.monitoring_params = config["dataset_params"]
        self.source = data_source
        self.dataset = None

    def _create_initial_dataframe(self):
        if isinstance(self.source, pd.DataFrame):
            dates_and_groups = [
                {DATE_COL: pair[0], GROUP_COL: pair[1]}
                for pair in product(
                    self.source[DATE_COL].unique(), self.source[GROUP_COL].unique()
                )
            ]
            result_df = pd.DataFrame(dates_and_groups).sort_values(
                by=[DATE_COL, GROUP_COL]
            )
            result_df[DATE_COL] = pd.to_datetime(result_df[DATE_COL])

        elif isinstance(self.source, str):
            if tf is not None:
                min_max_date = tf.gp_to_df(
                    f"select min({DATE_COL}) as min_date, max({DATE_COL}) as max_date from {self.source}"
                )
                dates = pd.date_range(
                    min_max_date["min_date"].values[0],
                    min_max_date["max_date"].values[0],
                )
                group_names = tf.gp_to_df(
                    f"select distinct {GROUP_COL} from {self.source}"
                )[GROUP_COL].values
                dates_and_groups = [
                    {DATE_COL: pair[0], GROUP_COL: pair[1]}
                    for pair in product(dates, group_names)
                ]
                result_df = pd.DataFrame(dates_and_groups).sort_values(
                    by=[DATE_COL, GROUP_COL]
                )
            else:
                dates_and_groups = []
                result_df = pd.DataFrame(columns=[DATE_COL, GROUP_COL])
        else:
            raise ValueError("Incorrect source type")

        return dates_and_groups, result_df

    def prepare_metrics_dataset(self) -> pd.DataFrame:
        dates_and_groups, result_df = self._create_initial_dataframe()

        if len(self.monitoring_params["metric_per_time"]) != 0:
            metrics_per_time_df = self._calculate_metrics_per_time(dates_and_groups)
            result_df = result_df.merge(
                metrics_per_time_df, on=[DATE_COL, GROUP_COL], how="left"
            )

        if len(self.monitoring_params["cumulative_metric_per_time"]) != 0:
            cum_metrics_per_time_df = self._calc_cum_metrics_per_time(dates_and_groups)
            result_df = result_df.merge(
                cum_metrics_per_time_df, on=[DATE_COL, GROUP_COL], how="left"
            )
        result_df = result_df.fillna(0).sort_values(by=[DATE_COL, GROUP_COL])
        self.dataset = result_df
        return result_df

    def _calculate_metrics_per_time(self, dates_and_groups: List[Dict[str, str]]):
        result_df = pd.DataFrame(dates_and_groups, columns=[DATE_COL, GROUP_COL])
        metrics_per_date = self.monitoring_params["metric_per_time"]
        if "new_users_cnt" in metrics_per_date.get(USER_COL):
            df_with_new_users = self.processer.calc_stats(
                self.source,
                {USER_COL: ["nunique"]},
                [FIRST_DATE_IN_TEST, GROUP_COL],
                {f"{USER_COL}_nunique": "new_users_cnt", FIRST_DATE_IN_TEST: DATE_COL},
            )
            if not df_with_new_users.empty:
                result_df = result_df.merge(
                    df_with_new_users, on=[DATE_COL, GROUP_COL], how="left"
                )

        calc_dict_per_date = {}
        for metric in metrics_per_date:
            if metric != USER_COL:
                calc_dict_per_date[metric] = metrics_per_date[metric]
            else:
                funcs = []
                for agg_func in metrics_per_date[metric]:
                    if agg_func != "new_users_cnt":
                        funcs.append(agg_func)

                if len(funcs) != 0:
                    calc_dict_per_date[metric] = funcs

        metrics_per_date_df = self.processer.calc_stats(
            self.source, calc_dict_per_date, [DATE_COL, GROUP_COL]
        )
        if not metrics_per_date_df.empty:
            result_df = result_df.merge(
                metrics_per_date_df, on=[DATE_COL, GROUP_COL], how="left"
            )

        return result_df

    def _calc_cum_metrics_per_time(self, dates_and_groups: List[Dict[str, str]]):
        result_df = pd.DataFrame(dates_and_groups, columns=[DATE_COL, GROUP_COL])
        cum_metrics_per_date = self.monitoring_params["cumulative_metric_per_time"]
        cum_bin_metrics = {}
        cum_cont_metrics = {}
        cum_user_metrics = {}
        for metric in cum_metrics_per_date:
            if metric in self.metrics["binary"]:
                cum_bin_metrics[metric] = cum_metrics_per_date[metric]
            elif metric != USER_COL and metric in self.metrics["continuous"]:
                cum_cont_metrics[metric] = cum_metrics_per_date[metric]
            elif metric == USER_COL:
                cum_user_metrics[USER_COL] = ['nunique']
            else:
                raise ValueError(
                    f"{metric} отсутствует в непрерывных и бинарных метриках или вы некорректно назвали поле с пользователями"
                )

        if len(cum_bin_metrics) != 0:
            cum_bin_metrics_df = self.processer.calc_cum_stats_per_date(
                self.source,
                cum_bin_metrics,
                group_columns=[DATE_COL, GROUP_COL],
                binary_metrics_flag=True,
            )
            if not cum_bin_metrics_df.empty:
                result_df = result_df.merge(
                    cum_bin_metrics_df, on=[DATE_COL, GROUP_COL], how="left"
                )

        if len(cum_cont_metrics) != 0:
            cum_cont_metrics_df = self.processer.calc_cum_stats_per_date(
                self.source, cum_cont_metrics, group_columns=[DATE_COL, GROUP_COL]
            )
            if not cum_cont_metrics_df.empty:
                result_df = result_df.merge(
                    cum_cont_metrics_df, on=[DATE_COL, GROUP_COL], how="left"
                )

        if len(cum_user_metrics) != 0:
            cum_new_users_df = self.processer.calc_cum_stats_per_date(
                self.source,
                cum_user_metrics,
                group_columns=[FIRST_DATE_IN_TEST, GROUP_COL],
            )
            if not cum_new_users_df.empty:
                result_df = result_df.merge(
                    cum_new_users_df, on=[DATE_COL, GROUP_COL], how="left"
                )

        return result_df

    def prepare_test_results_dataset(self) -> pd.DataFrame:
        if len(self.test_params["metrics"]) == 0:
            return pd.DataFrame()
        
        (
            updated_metrics,
            metrics_calcs,
            dates,
            groups,
        ) = self._aggregate_dataset_for_test_calc_by_user_dataset()

        test_calculator = TestCalculator(
                updated_metrics, self.test_params, self.processer
            )
        
        if isinstance(self.source, str) and len(self.test_params["metrics"]) != 0:
            if self.test_params["criterion"] != "Т-тест":
                raise ValueError(
                    "Данный тип статистического критерий не поддерживается. Установите Т-тест"
                )
            
            test_results = test_calculator.calculate_test_results_per_days_and_stats(self.source, self.test_params['metrics'], groups)
            return test_results

        if isinstance(self.source, pd.DataFrame):
            test_result_df = test_calculator.calculate_test_results_per_days_and_users(
                self.source, dates, groups, metrics_calcs
            )
            return test_result_df

    def _aggregate_dataset_for_test_calc_by_user_dataset(self):
        updated_metrics = {
            "continuous": [],
            "binary": [],
        }
        metrics_calcs = {}

        for metric in self.test_params["metrics"]:
            if metric in self.metrics["continuous"]:
                updated_metrics["continuous"].append(metric)
                metrics_calcs[metric] = ["sum"]
            elif metric in self.metrics["binary"]:
                updated_metrics["binary"].append(metric)
                metrics_calcs[metric] = ["max"]
            else:
                raise ValueError(
                    'Неизвестный тип у "{metric}". Отнесите ее или к непрерывным или к бинарным'
                )

        if self.dataset is not None:
            dates = self.dataset[DATE_COL].unique()
            groups = self.dataset[GROUP_COL].unique()
        else:
            _, result_df = self._create_initial_dataframe()
            dates = result_df[DATE_COL].unique()
            groups = result_df[GROUP_COL].unique()

        return updated_metrics, metrics_calcs, dates, groups
