from typing import Tuple
import ambrosia.tools.bin_intervals as bin_pkg
import ambrosia.tools.pvalue_tools as pvalue_pkg
from ambrosia import types
import numpy as np
import scipy.stats as sps


def binary_absolute_result(
    success_a: int, 
    success_b: int,
    trials_a: int,
    trials_b: int,
    alpha: float,
    **kwargs
) -> types._SubResultType:
    pvalue: float = bin_pkg.BinomTwoSampleCI.calculate_pvalue(
        a_success=success_a, b_success=success_b, a_trials=trials_a, b_trials=trials_b, **kwargs
    )
    conf_intervals = bin_pkg.BinomTwoSampleCI.confidence_interval(
        a_success=success_a,
        b_success=success_b,
        a_trials=trials_a,
        b_trials=trials_b,
        confidence_level=1 - alpha,
        **kwargs,
    )
    return {
        "first_type_error": alpha,
        "pvalue": pvalue,
        "confidence_interval": conf_intervals,
    }


def ttest_absolut_result(
        mean_a: float,
        std_a: float,
        nobs_a: int,
        mean_b: float,
        std_b: float,
        nobs_b: int,
        alpha: np.ndarray = np.array([0.05]),
        equal_var: bool = False,
        alternative: str = 'two-sided'
):
    pvalue = sps.ttest_ind_from_stats(
        mean1=mean_a,
        std1=std_a,
        nobs1=nobs_a,
        mean2=mean_b,
        std2=std_b,
        nobs2=nobs_b,
        equal_var=equal_var,
        alternative=alternative
    ).pvalue
    conf_intervals = build_interval(
        mean_b - mean_a,
        std_a ** 2,
        std_b ** 2,
        nobs_a,
        nobs_b,
        alpha,
        alternative
    )
    return {
        "first_type_error": alpha,
        "pvalue": pvalue,
        "confidence_interval": conf_intervals,
    }

def build_interval(
        center: float,
        var_a: float,
        var_b: float,
        nobs_a: int,
        nobs_b: int,
        alpha: np.array = np.array([0.05]),
        alternative: str = 'two-sided'
):
    quantiles, std_error = get_ttest_info_from_stats(var_a, var_b, nobs_a, nobs_b, alpha)
    left_ci: np.ndarray = center - quantiles * std_error
    right_ci: np.ndarray = center + quantiles * std_error
    left_ci, right_ci = pvalue_pkg.choose_from_bounds(left_ci, right_ci, alternative)
    conf_intervals = list(zip(left_ci, right_ci))
    return conf_intervals


def get_ttest_info_from_stats(
    var_a: float, var_b: float, n_obs_a: int, n_obs_b: int, alpha: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns quantiles and standard deviation of Ttest criterion statistic
    """
    compound_se: float = np.sqrt(var_a / n_obs_a + var_b / n_obs_b)
    denominator: float = (var_a / n_obs_a) ** 2 / (n_obs_a - 1) + (var_b / n_obs_b) ** 2 / (n_obs_b - 1)
    dim: float = compound_se**2 / denominator
    quantiles: np.ndarray = sps.t.ppf(1 - alpha / 2, df=dim)
    return quantiles, compound_se
