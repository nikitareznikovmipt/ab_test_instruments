from itertools import product
from typing import Dict, List

import pandas as pd

from ..config import DATE_COL, FIRST_DATE_IN_TEST, USER_COL

AVAILEBLE_AGGREGATES = ["count", "mean", "nunique", "std", "max", "min"]


class Pandas_Processer:
    def calc_stats(
        self,
        df: pd.DataFrame,
        group_columns: List[str],
        metrics: Dict[str, List[str]],
        renaming_dict: Dict[str, str] = {},
    ) -> pd.DataFrame:
        if len(group_columns) == 0:
            res = df.agg(metrics)
            res_values = {}

            for index, col in product(res.index, res.columns):
                val = res.loc[index, col]
                if not pd.isna(val):
                    res_values[f"{col}_{index}"] = val

            agg_df = pd.DataFrame([res_values])
        else:
            agg_df = df.groupby(group_columns, as_index=False).agg(metrics)

        if isinstance(agg_df.columns, pd.MultiIndex):
            agg_df.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns
            ]
        else:
            agg_df.columns = [
                f"{col}_{metrics[col]}" if col in metrics else col
                for col in agg_df.columns
            ]

        agg_df.rename(columns=renaming_dict, inplace=True)
        return agg_df

    def calc_cum_stats_per_date(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, List[str]],
        group_columns: List[str] = [],
        renaming_dict: Dict[str, str] = {},
        binary_metrics_flag: bool = False,
    ) -> pd.DataFrame:
        cum_df = pd.DataFrame()

        if DATE_COL not in group_columns and FIRST_DATE_IN_TEST not in group_columns:
            raise ValueError("Передай одну из колонок с датой")

        curr_date_col = DATE_COL if DATE_COL in group_columns else FIRST_DATE_IN_TEST
        group_columns.remove(curr_date_col)

        dates = df[curr_date_col].sort_values().unique()
        cum_renaming_dict = {
            f"{metric}_{agg_func}": f"cum_{metric}_{agg_func}"
            for metric in metrics
            for agg_func in metrics[metric]
        }
        for date in dates:
            df_for_calc = df[df[curr_date_col] <= date]
            if binary_metrics_flag:
                precalc_metrics = {metric: ["max"] for metric in metrics}
                precalc_renaming = {f"{metric}_max": metric for metric in metrics}
                precalc_df = self.calc_stats(
                    df_for_calc,
                    [USER_COL, *group_columns],
                    precalc_metrics,
                    precalc_renaming,
                )
                result = self.calc_stats(
                    precalc_df, group_columns, metrics, cum_renaming_dict
                )

            else:
                result = self.calc_stats(
                    df_for_calc, group_columns, metrics, cum_renaming_dict
                )
            corr_order_wo_date = [*result.columns]
            result[curr_date_col] = date
            corr_order = [curr_date_col, *corr_order_wo_date]
            cum_df = pd.concat([cum_df, result[corr_order]], ignore_index=True)

        cum_df.rename(columns=renaming_dict, inplace=True)
        cum_df.rename(columns={curr_date_col: DATE_COL}, inplace=True)
        return cum_df
