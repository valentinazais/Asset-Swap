from __future__ import annotations

import numpy as np
import pandas as pd


def build_discount_factor(tenors: list[float], zero_rates: list[float]):
    """Continuous-compounding zero curve with linear interpolation on rates.

    Parameters
    ----------
    tenors : maturities in years (ascending).
    zero_rates : zero rates in decimal (e.g. 0.03 for 3%).

    Returns
    -------
    Callable t -> DF(t).
    """
    ts = np.asarray(tenors, dtype=float)
    rs = np.asarray(zero_rates, dtype=float)

    def DF(t: float) -> float:
        if t <= 0.0:
            return 1.0
        r = float(np.interp(t, ts, rs))
        return float(np.exp(-r * t))

    return DF


def forward_simple(DF, t1: float, t2: float) -> float:
    """Simple-compounded forward between t1 and t2 (ACT-like, tau = t2 - t1)."""
    tau = t2 - t1
    return (DF(t1) / DF(t2) - 1.0) / tau


def curve_dataframe(tenors: list[float], zero_rates: list[float],
                    n_points: int = 100) -> pd.DataFrame:
    """Dense curve for plotting."""
    ts = np.linspace(min(tenors), max(tenors), n_points)
    rs = np.interp(ts, tenors, zero_rates)
    return pd.DataFrame({"tenor": ts, "zero_rate": rs})
