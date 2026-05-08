from __future__ import annotations

import numpy as np
import pandas as pd

from asw.bond import coupon_schedule
from asw.curve import forward_simple
from asw.pricing import par_asw_spread


def cashflow_table(coupon_rate: float, maturity: float, fix_frequency: int,
                    flt_frequency: int, spread: float, DF,
                    notional: float = 100.0) -> pd.DataFrame:
    """Per-period cashflows for the swap legs (investor perspective)."""
    fix_dates = set(np.round(coupon_schedule(maturity, fix_frequency), 8))
    flt_dates = coupon_schedule(maturity, flt_frequency)
    tau_fix = 1.0 / fix_frequency
    tau_flt = 1.0 / flt_frequency

    rows = []
    t_prev = 0.0
    for t in flt_dates:
        L = forward_simple(DF, t_prev, t)
        cf_flt = -(L + spread) * notional * tau_flt
        cf_fix = (coupon_rate * notional * tau_fix
                  if round(t, 8) in fix_dates else 0.0)
        rows.append({
            "t": t,
            "DF": DF(t),
            "libor_fwd": L,
            "cf_fix_received": cf_fix,
            "cf_flt_paid": cf_flt,
            "cf_net": cf_fix + cf_flt,
        })
        t_prev = t

    return pd.DataFrame(rows)


def price_sensitivity(coupon_rate: float, maturity: float,
                       fix_frequency: int, flt_frequency: int, DF,
                       price_min: float = 90.0, price_max: float = 110.0,
                       n: int = 41, notional: float = 100.0) -> pd.DataFrame:
    prices = np.linspace(price_min, price_max, n)
    spreads = [par_asw_spread(coupon_rate, maturity, p, fix_frequency,
                               flt_frequency, DF, notional) * 1e4
               for p in prices]
    return pd.DataFrame({"dirty_price": prices, "spread_bps": spreads})


def maturity_sensitivity(coupon_rate: float, dirty_price: float,
                          fix_frequency: int, flt_frequency: int, DF,
                          mat_min: float = 1.0, mat_max: float = 10.0,
                          n: int = 19, notional: float = 100.0) -> pd.DataFrame:
    mats = np.linspace(mat_min, mat_max, n)
    spreads = [par_asw_spread(coupon_rate, m, dirty_price, fix_frequency,
                               flt_frequency, DF, notional) * 1e4
               for m in mats]
    return pd.DataFrame({"maturity": mats, "spread_bps": spreads})
