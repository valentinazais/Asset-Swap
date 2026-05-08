from __future__ import annotations

import numpy as np


def coupon_schedule(maturity: float, frequency: int) -> list[float]:
    """Coupon dates in years from t=0 (regular schedule, no stub)."""
    n = int(round(maturity * frequency))
    dt = 1.0 / frequency
    return [(i + 1) * dt for i in range(n)]


def dirty_price_from_ytm(coupon_rate: float, maturity: float, ytm: float,
                          frequency: int, notional: float = 100.0) -> float:
    """Dirty price assuming YTM compounded `frequency` times per year."""
    dates = coupon_schedule(maturity, frequency)
    c = coupon_rate * notional / frequency
    pv = 0.0
    for t in dates:
        pv += c / (1.0 + ytm / frequency) ** (t * frequency)
    pv += notional / (1.0 + ytm / frequency) ** (maturity * frequency)
    return pv


def ytm_from_dirty_price(dirty: float, coupon_rate: float, maturity: float,
                         frequency: int, notional: float = 100.0) -> float:
    """Invert dirty price to YTM via Brent."""
    from scipy.optimize import brentq

    def diff(y: float) -> float:
        return dirty_price_from_ytm(coupon_rate, maturity, y, frequency, notional) - dirty

    return brentq(diff, -0.5, 1.0, xtol=1e-10)
