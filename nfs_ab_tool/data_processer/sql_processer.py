from typing import Dict, List

from ..config import DATE_COL, FIRST_DATE_IN_TEST, GROUP_COL, USER_COL

AVAILEBLE_AGGREGATES = ["count", "mean", "nunique", "std", "max", "min"]

MAPPING_AGG_FUNCS = {
    "sum": "SUM(",
    "count": "COUNT(",
    "mean": "AVG(",
    "nunique": "COUNT(DISTINCT ",
    "std": "STDDEV(",
    "min": "MIN(",
    "max": "MAX(",
}


class SQL_Processer:
    def calc_stats(
        self,
        table_name: str,
        group_columns: List[str],
        metrics: Dict[str, List[str]],
        renaming_dict: Dict[str, str] = {},
    ) -> str:
        renamed_group_columns = [
            f"{name} as {renaming_dict.get(name)}" if renaming_dict.get(name) else name
            for name in group_columns
        ]
        selected_columns = renamed_group_columns.copy()

        for metric in metrics:
            for agg_func in metrics[metric]:
                metric_name = f"{metric}_{agg_func}"
                metric_name = renaming_dict.get(metric_name, metric_name)
                selected_columns.append(
                    f"{MAPPING_AGG_FUNCS[agg_func]}{metric}) as {metric_name}"
                )

        query = f"""
        SELECT
            {", ".join(selected_columns)}
        FROM {table_name}
        GROUP BY {','.join(map(str, list(range(1, len(group_columns) + 1))))}
        """

        return query

    def calc_cum_stats_per_date(
        self,
        table_name: str,
        metrics: Dict[str, List[str]],
        group_columns: List[str] = None,
        renaming_dict: Dict[str, str] = None,
        binary_metrics_flag: bool = False,
        renaming_group_columns: str = None,
        groups_filter: str = None,
    ) -> str:
        if group_columns is None:
            group_columns = []

        if renaming_dict is None:
            renaming_dict = {}

        if DATE_COL not in group_columns and FIRST_DATE_IN_TEST not in group_columns:
            raise ValueError("Передай одну из колонок с датой")

        group_col = f"{GROUP_COL if renaming_group_columns is None else renaming_group_columns} AS {GROUP_COL}"

        # ДЛЯ ЮЗЕРОВ ЗАХАРКОЖЕНА ГРУППИРОВКА ПО ГРУППЕ И ДАТЕ
        if "user_id" in metrics:
            if "new_users_cnt" in metrics.get("user_id", []):
                return f"""
                with precalc as (
                    SELECT
                        {group_col}, 
                        {FIRST_DATE_IN_TEST} AS {DATE_COL},
                        COUNT(DISTINCT {USER_COL}) as uniq_cnt
                    FROM {table_name}
                    WHERE 1=1
                    {'AND ' + groups_filter if groups_filter is not None else ''}
                    GROUP BY 1,2
                )
                SELECT
                    {GROUP_COL},
                    {DATE_COL},
                    SUM(uniq_cnt) OVER(PARTITION BY {GROUP_COL} OVER {DATE_COL}) as cum_new_users_cnt
                FROM precalc
                """
            elif "nunique" in metrics.get("user_id", []):
                return f""""
                SELECT
                    {group_col},
                    {DATE_COL},
                    COUNT(DISTINCT user_id) as cum_users_cnt
                FROM {table_name}
                WHERE 1=1
                {'AND ' + groups_filter if groups_filter is not None else ''}
                GROUP BY 1,2
                """
            else:
                raise ValueError(
                    "Под ключем user_id переданы некорректные аггрегирующие функции, поддерживаются только nunique и new_users_cnt"
                )

        selected_precalc_columns = [DATE_COL, USER_COL, group_col]
        for metric in metrics:
            if not binary_metrics_flag and metric != "user_id":
                agg_func = "sum"
            else:
                agg_func = "max"

            selected_precalc_columns.append(
                self._create_window_metric(
                    metric, agg_func, [GROUP_COL, USER_COL], metric, DATE_COL
                )
            )
        precalc_query = f"""
        with precalc as (
            SELECT
                {','.join(selected_precalc_columns)}
            FROM {table_name}
        """

        if groups_filter is not None:
            precalc_query += f"    WHERE {groups_filter})\n"
        else:
            precalc_query += ")\n"

        query = self.calc_stats("precalc", group_columns, metrics, renaming_dict)

        result_query = precalc_query + query

        return result_query

    def _create_window_metric(
        self,
        initial_metric_name: str,
        agg_func: str,
        group_columns: List[str],
        last_name: str,
        date_col: str,
    ):
        simple_funcs = ["sum", "max"]
        if agg_func not in simple_funcs:
            raise ValueError(
                f'Кумулятивная метрика рассчитывается только по {",".join(simple_funcs)}'
            )

        metric_str = f"{MAPPING_AGG_FUNCS[agg_func]}{initial_metric_name}) OVER("
        if len(group_columns) != 0:
            metric_str += f"PARTITION BY {','.join(group_columns)} "

        metric_str += f"ORDER BY {date_col}) as {last_name}"

        return metric_str
