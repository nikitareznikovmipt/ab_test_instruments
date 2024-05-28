import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import pytest

from nfs_ab_tool.data_processer.sql_processer import SQL_Processer


@pytest.fixture
def sql_processor():
    return SQL_Processer()


def query_transforms(query):
    return query.strip().lower().replace("\n", "").replace(" ", "")


def test_calc_stats(sql_processor):
    table_name = "my_table"
    group_columns = ["category"]
    metrics = {"sales": ["sum", "mean"], "quantity": ["count", "max"]}
    query = sql_processor.calc_stats(table_name, group_columns, metrics)

    expected_query = """
        SELECT
            category,
            SUM(sales) as sales_sum,
            AVG(sales) as sales_mean,
            COUNT(quantity) as quantity_count,
            MAX(quantity) as quantity_max
        FROM my_table
        GROUP BY 1
    """

    assert query_transforms(query) == query_transforms(expected_query), query


def test_calc_cum_stats_per_date_invalid_input(sql_processor):
    table_name = "my_table"
    metrics = {"sales": ["sum", "mean"]}
    with pytest.raises(ValueError):
        sql_processor.calc_cum_stats_per_date(table_name, metrics)

    with pytest.raises(ValueError):
        sql_processor.calc_cum_stats_per_date(table_name, metrics, group_columns=[])

    with pytest.raises(ValueError):
        sql_processor.calc_cum_stats_per_date(
            table_name, metrics, group_columns=["other_column"]
        )
