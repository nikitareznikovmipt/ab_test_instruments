import re
from itertools import combinations
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from ambrosia.tester import test

from nfs_ab_tool.data_processer import Data_Processer
from .statistic_tools import binary_absolute_result, ttest_absolut_result

from ..config import DATE_COL, GROUP_COL, USER_COL


AVAILABLE_METRICS_TYPES = ["continuous", "binary"]
METHODS_MAPPING = {
    "Бутстрап": {"method": "empiric"},
    "Т-тест": {"criterion": "ttest"},
    "Манн-Уитни": {"criterion": "mw"},
}
ALTERNATIVES = ["two-sided", "less", "greater"]


class TestCalculator:
    def __init__(
        self,
        metrics: Dict[str, List[str]],
        test_config: Dict,
        processer: Data_Processer,
    ) -> None:
        self.config = self._check_and_correct_config(test_config)

        if self._check_metrics(metrics):
            self.metrics = metrics

        self.processer = processer

    def _check_and_correct_config(self, config: Dict) -> Dict:
        """
        Check config
        """
        if "first_type_errors" not in config:
            raise ValueError("Необходимо зафиксировать вероятность ошибки I рода")

        if "alternative" in config:
            if config["alternative"] not in ALTERNATIVES:
                raise ValueError(
                    f"Значение alternative должно быть одно из списка: {ALTERNATIVES}"
                )
            alternative = config["alternative"]
        else:
            alternative = "two-sided"

        if "second_type_errors" not in config:
            raise ValueError("Необходимо зафиксировать вероятность ошибки II рода")

        if "criterion" not in config:
            correct_criterion = {"criterion": "ttest"}
        else:
            if config["criterion"] not in METHODS_MAPPING:
                raise ValueError(
                    f"Критерий задан некорректно или еще недоступен для использования. \
                    Досупные криетрии: {list(METHODS_MAPPING.keys())}"
                )
            correct_criterion = METHODS_MAPPING[config["criterion"]]

        correct_config = {
            "first_type_errors": config["first_type_errors"],
            "second_type_errors": config["second_type_errors"],
            "alternative": alternative,
            "criterion": correct_criterion,
        }

        return correct_config

    def _check_metrics(self, metrics_dict: Dict[str, List[str]]) -> bool:
        """
        Check metric types
        """
        for key in metrics_dict:
            if key not in AVAILABLE_METRICS_TYPES:
                raise ValueError(
                    f"Названия метрик не переданы или переданы некорректные типы. \
                        Список корректных типов: {AVAILABLE_METRICS_TYPES}"
                )

        return True

    def calc_stats_by_metric(
        self, data: pd.DataFrame, as_table: bool = True
    ) -> Union[pd.DataFrame, Dict]:
        """
        Function for test evaluation

        Args:
            data_per_users (pd.DataFrame): Dataframe with users, metric aggregations, group
            as_table (bool, optional): returned format.
                                       If as_table is True return pd.DataFrame, else Dict.
                                       Defaults to True.
        Returns:
            Union[pd.DataFrame, Dict]: test tesults in table
        """

        if GROUP_COL not in data.columns:
            raise ValueError(
                f"Колонка {GROUP_COL} с группой остутствует в переданной таблице"
            )
        metrics_list = [
            metric_name
            for metrics in self.metrics
            for metric_name in self.metrics[metrics]
        ]

        for metric in metrics_list:
            if metric not in data.columns:
                raise ValueError(
                    f"Колонка {metric} с метрикой остутствует в переданной таблице"
                )

        return self._test_results(data, as_table)

    def _test_results(
        self, data: pd.DataFrame, as_table: bool = True
    ) -> Union[pd.DataFrame, Dict]:
        """
        Calculate test statistics
        """

        if "binary" in self.metrics:
            binary_results = test(
                dataframe=data,
                metrics=self.metrics["binary"],
                column_groups=GROUP_COL,
                first_type_errors=self.config["first_type_errors"],
                correction_method=None,
                alternative=self.config["alternative"],
                as_table=False,
            )
        else:
            binary_results = []

        if "continuous" in self.metrics:
            continuous_results = test(
                dataframe=data,
                metrics=self.metrics["continuous"],
                column_groups=GROUP_COL,
                first_type_errors=self.config["first_type_errors"],
                as_table=False,
                correction_method=None,
                alternative=self.config["alternative"],
                **self.config["criterion"],
            )
        else:
            continuous_results = []

        test_results = [*binary_results, *continuous_results]

        for test_result in test_results:
            test_result["mean in A"] = data[
                data[GROUP_COL] == test_result["group A label"]
            ][test_result["metric name"]].mean()

            test_result["uplift, %"] = (
                test_result["effect"] * 100 / test_result["mean in A"]
            )

            test_result["num in A"] = data[
                data[GROUP_COL] == test_result["group A label"]
            ].shape[0]

            test_result["mean in B"] = data[
                data[GROUP_COL] == test_result["group B label"]
            ][test_result["metric name"]].mean()

            test_result["num in B"] = data[
                data[GROUP_COL] == test_result["group B label"]
            ].shape[0]

            test_result["is_success"] = (
                test_result["pvalue"] < test_result["first_type_error"]
            )

            test_result["percentage cnt in A"] = round(
                test_result["num in A"]
                * 100
                / (test_result["num in B"] + test_result["num in A"]),
                1,
            )
            test_result["percentage cnt in B"] = round(
                test_result["num in B"]
                * 100
                / (test_result["num in B"] + test_result["num in A"]),
                1,
            )

        if as_table:
            return pd.DataFrame(test_results).rename(columns={"effect": "abs_effect"})

        return test_results

    @staticmethod
    def _groups_for_compare(groups: List[str]):
        letters_with_digits = {}
        groups_for_compare = []
        for group in groups:
            match = re.match(r"([a-zA-Z]?)(\d?)", group)
            if match:
                letter = match.group(1)
                digit = match.group(2)
                if not re.match("^[a-zA-Z]$", letter):
                    raise ValueError("Incorrect groups name")

                if letter in letters_with_digits:
                    letters_with_digits[letter].append(digit)
                else:
                    letters_with_digits[letter] = [digit]

        for letter in letters_with_digits:
            correct_indexes = []
            for elem in letters_with_digits[letter]:
                if re.match("^\d$", elem):
                    correct_indexes.append(elem)
            letters_with_digits[letter] = correct_indexes
            if len(correct_indexes) > 1:
                for index_A, index_B in combinations(correct_indexes, 2):
                    groups_for_compare.append(
                        {
                            "A": letter + index_A,
                            "B": letter + index_B,
                            "renaming": {},
                            "need_correction": False,
                        }
                    )

        for group_A_col, group_B_col in combinations(letters_with_digits.keys(), 2):
            renaming_dict = {}
            if len(letters_with_digits[group_A_col]) > 0:
                for index in letters_with_digits[group_A_col]:
                    renaming_dict[group_A_col + index] = "A"
            else:
                renaming_dict[group_A_col] = group_A_col
            if len(letters_with_digits[group_B_col]) > 0:
                for index in letters_with_digits[group_B_col]:
                    renaming_dict[group_B_col + index] = "B"
            else:
                renaming_dict[group_B_col] = group_B_col

            groups_for_compare.append(
                {
                    "A": group_A_col,
                    "B": group_B_col,
                    "renaming": renaming_dict,
                    "need_correction": True,
                }
            )
        return groups_for_compare

    def calculate_test_results_per_days_and_users(
        self,
        data: pd.DataFrame,
        dates: List[np.datetime64],
        groups: List[str],
        metrics_agg: Dict[str, List[str]],
    ):
        groups_for_compare = self._groups_for_compare(groups=groups)
        renaming_dict = {
            f"{metric}_{agg_func}": metric
            for metric in metrics_agg
            for agg_func in metrics_agg[metric]
        }

        renaming_dict = {
            f"{metric}_{agg_func}": metric
            for metric in metrics_agg
            for agg_func in metrics_agg[metric]
        }

        test_result_df = pd.DataFrame()
        for groups in groups_for_compare:
            source_copy = data.copy()
            source_copy = source_copy[
                source_copy[GROUP_COL].isin(
                    list(groups["renaming"].keys()) + [groups["A"], groups["B"]]
                )
            ]
            if len(groups["renaming"]) != 0:
                source_copy[GROUP_COL] = source_copy[GROUP_COL].map(groups["renaming"])
            for date in dates:
                try:
                    dataset_with_calc_metrics = self.processer.calc_stats(
                        source_copy[source_copy[DATE_COL] <= date],
                        metrics=metrics_agg,
                        group_columns=[USER_COL, GROUP_COL],
                        renaming_dict=renaming_dict,
                    )

                    curr_test_res = self.calc_stats_by_metric(dataset_with_calc_metrics)
                    curr_test_res[DATE_COL] = date
                    test_result_df = pd.concat([test_result_df, curr_test_res])
                except ValueError:
                    print(
                        f"Cant calculate test result for {date} and A = {groups['A']}, B = {groups['B']}"
                    )

        return test_result_df

    def calculate_test_results_per_days_and_stats(
        self, source: str, metrics: List[str], groups: List[str]
    ):
        statistic_df = self._prepare_df(source, metrics, groups)
        return self._calc_test_days_per_days(statistic_df)
    
    def _calc_test_days_per_days(self, statistic_df: pd.DataFrame):
        results_with_stats = []
        for row in statistic_df.to_dict(orient='records'):
            if row['metric name'] in self.metrics['binary']:
                stats_res = binary_absolute_result(
                    int(row['mean in A'] * row['num in A']),
                    int(row['mean in B'] * row['num in B']),
                    row['num in A'],
                    row['num in B'],
                    alpha=self.config['first_type_errors'][0],
                    alternative=self.config['alternative']
                )
                row.update(stats_res)
                results_with_stats.append(row)
            elif row['metric name'] in self.metrics['continuous']:
                stats_res = ttest_absolut_result(
                    row['mean in A'],
                    row['std in A'],
                    row['num in A'],
                    row['mean in B'],
                    row['std in B'],
                    row['num in B'],
                    alpha=self.config['first_type_errors']
                )
                row.update(stats_res)
                results_with_stats.append(row)
            else:
                raise ValueError(
                    f"{row['metric name']} не относиться ни к непрерывной, ни к бинарной метрике"
                )
        
        return pd.DataFrame(results_with_stats)

    def _prepare_df(
        self, source: str, metrics: List[str], groups: List[str]
    ) -> pd.DataFrame:
        groups_for_compare = self._groups_for_compare(groups=groups)
        print(groups_for_compare)

        binary_metrics = []
        continuous_metrics = []

        for metric in metrics:
            if metric in self.metrics["binary"]:
                binary_metrics.append(metric)
            elif metric in self.metrics["continuous"]:
                continuous_metrics.append(metric)
            else:
                raise ValueError(
                    f"{metric} не относиться ни к непрерывной, ни к бинарной метрике"
                )

        stats_df = pd.DataFrame()

        for group in groups_for_compare:
            group_for_filter = list(
                set(list(group["renaming"].keys()) + [group["A"], group["B"]])
            )
            group_filter = f"""{GROUP_COL} in ({','.join(map(lambda x: f"'{x}'", group_for_filter))})"""

            if len(group["renaming"]) != 0:
                renaming_group = "case "
                for elem in group["renaming"]:
                    renaming_group += f"""when {GROUP_COL} = '{elem}' then '{group["renaming"][elem]}' """
                renaming_group += f"else {GROUP_COL} end"
            else:
                renaming_group = None

            # Посчитать количество юзеров в группе
            users_df = self.processer.calc_cum_stats_per_date(
                data=source,
                metrics={"user_id": ["nunique"]},
                group_columns=[DATE_COL, GROUP_COL],
                renaming_group_columns=renaming_group,
                groups_filter=group_filter,
            )
            transformed_user_df = self._transform_df(
                users_df, metrics={"user_id": ["nunique"]}, group_info=group
            )

            if len(binary_metrics) != 0:
                metrics = {elem: ["mean", "std"] for elem in binary_metrics}
                binary_df = self.processer.calc_cum_stats_per_date(
                    data=source,
                    metrics=metrics,
                    group_columns=[DATE_COL, GROUP_COL],
                    binary_metrics_flag=True,
                    renaming_group_columns=renaming_group,
                    groups_filter=group_filter,
                )
                transformed_binary_df = self._transform_df(
                    binary_df, metrics=metrics, group_info=group
                )
            else:
                transformed_binary_df = pd.DataFrame()

            if len(continuous_metrics) != 0:
                metrics = {elem: ["mean", "std"] for elem in continuous_metrics}
                continuous_df = self.processer.calc_cum_stats_per_date(
                    data=source,
                    metrics=metrics,
                    group_columns=[DATE_COL, GROUP_COL],
                    renaming_group_columns=renaming_group,
                    groups_filter=group_filter,
                )
                transformed_continuous_df = self._transform_df(
                    continuous_df, metrics=metrics, group_info=group
                )
            else:
                transformed_continuous_df = pd.DataFrame()
            
            if transformed_continuous_df.empty:
                transformed_user_df = transformed_user_df.merge(transformed_binary_df, on=[DATE_COL, 'group A label', 'group B label'])
            elif transformed_binary_df.empty:
                transformed_user_df = transformed_user_df.merge(transformed_continuous_df, on=[DATE_COL, 'group A label', 'group B label'])
            else:
                merged_metrics = transformed_continuous_df.merge(transformed_binary_df, on=[DATE_COL, 'group A label', 'group B label', 'metric name'])
                transformed_user_df = transformed_user_df.merge(merged_metrics, on=[DATE_COL, 'group A label', 'group B label'])

            stats_df = pd.concat([stats_df, transformed_user_df])

        return stats_df

    def _transform_df(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, List[str]],
        group_info: Dict[str, str],
    ) -> pd.DataFrame:
        transformed_data = []
        uniq_dates = df[DATE_COL].unique()

        for date in uniq_dates:
            for metric in metrics:
                if "mean" in metrics[metric] and "std" in metrics[metric]:
                    mean_B = df[
                        (df[DATE_COL] == date) & (df[GROUP_COL] == group_info["B"])
                    ][f"{metric}_mean"].values[0]
                    mean_A = df[
                        (df[DATE_COL] == date) & (df[GROUP_COL] == group_info["A"])
                    ][f"{metric}_mean"].values[0]

                    transformed_data.append(
                        {
                            DATE_COL: date,
                            "group A label": group_info["A"],
                            "group B label": group_info["B"],
                            'metric name': metric,
                            "mean in A": mean_A,
                            "mean in B": mean_B,
                            "std in A": df[
                                (df[DATE_COL] == date)
                                & (df[GROUP_COL] == group_info["A"])
                            ][f"{metric}_std"].values[0],
                            "std in B": df[
                                (df[DATE_COL] == date)
                                & (df[GROUP_COL] == group_info["B"])
                            ][f"{metric}_std"].values[0],
                            "abs_effect": mean_B - mean_A,
                            "uplift, %": (mean_B - mean_A) * 100 / mean_A,
                        }
                    )

                elif "nunique" in metrics[metric]:
                    num_A = df[
                        (df[DATE_COL] == date) & (df[GROUP_COL] == group_info["A"])
                    ]["cum_users_cnt"].values[0]
                    num_B = df[
                        (df[DATE_COL] == date) & (df[GROUP_COL] == group_info["B"])
                    ]["cum_users_cnt"].values[0]

                    transformed_data.append(
                        {
                            DATE_COL: date,
                            "group A label": group_info["A"],
                            "group B label": group_info["B"],
                            "num in A": num_A,
                            "num in B": num_B,
                            "percentage cnt in A": num_A / (num_A + num_B),
                            "percentage cnt in B": num_B / (num_A + num_B),
                        }
                    )

        return pd.DataFrame(transformed_data)
