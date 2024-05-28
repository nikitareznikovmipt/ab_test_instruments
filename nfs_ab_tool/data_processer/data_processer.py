from typing import Dict, List, Union

import pandas as pd

try:
    import tinkoff as tf  # type: ignore
except ImportError:
    tf = None

from ..config import DATE_COL, GROUP_COL
from .pandas_processer import Pandas_Processer
from .sql_processer import SQL_Processer

AVAILEBLE_ENGINES = ["pandas", "gp"]
AVAILEBLE_AGGREGATES = ["count", "mean", "uniq", "std"]


class Data_Processer:
    def __init__(
        self,
        engine: str = "pandas",
    ) -> None:
        self.engine = engine

        if engine == "pandas":
            self.processer = Pandas_Processer()
        elif engine == "gp":
            self.processer = SQL_Processer()
        else:
            raise ValueError(
                f'Unknown engine: {engine}. Availeble engines: {",".join(AVAILEBLE_ENGINES)}'
            )

    def calc_stats(
        self,
        data: Union[str, pd.DataFrame],
        metrics: Dict[str, List[str]],
        group_columns: List[str],
        renaming_dict: Dict[str, str] = {},
    ) -> pd.DataFrame:
        if self.engine == "gp":
            query = self.processer.calc_stats(
                data,
                group_columns=group_columns,
                metrics=metrics,
                renaming_dict=renaming_dict,
            )
            if tf is not None:
                result = tf.gp_to_df(query, gp_service="gp")
            else:
                print(query)
                result = pd.DataFrame(columns=[DATE_COL, GROUP_COL])
        else:
            result = self.processer.calc_stats(
                data,
                group_columns=group_columns,
                metrics=metrics,
                renaming_dict=renaming_dict,
            )
        if DATE_COL in result.columns:
            result[DATE_COL] = pd.to_datetime(result[DATE_COL])
        return result

    def calc_cum_stats_per_date(
        self,
        data: Union[str, pd.DataFrame],
        metrics: Dict[str, List[str]],
        group_columns: List[str] = [],
        renaming_dict: Dict[str, str] = {},
        binary_metrics_flag: bool = False,
        renaming_group_columns: str = None,
        groups_filter: str = None,
    ) -> pd.DataFrame:
        if self.engine == "gp":
            query = self.processer.calc_cum_stats_per_date(
                data,
                group_columns=group_columns,
                metrics=metrics,
                renaming_dict=renaming_dict,
                binary_metrics_flag=binary_metrics_flag,
                renaming_group_columns=renaming_group_columns,
                groups_filter=groups_filter,
            )
            if tf is not None:
                result = tf.gp_to_df(query, gp_service="gp")
            else:
                print(query)
                return pd.DataFrame(columns=[DATE_COL, GROUP_COL])
        else:
            result = self.processer.calc_cum_stats_per_date(
                data,
                group_columns=group_columns,
                metrics=metrics,
                renaming_dict=renaming_dict,
                binary_metrics_flag=binary_metrics_flag,
            )

        if DATE_COL in result.columns:
            result[DATE_COL] = pd.to_datetime(result[DATE_COL])

        return result
