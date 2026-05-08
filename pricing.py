import numpy as np


def build_curve(tenors, zero_rates):
    """
    tenors: list[float] maturités en années
    zero_rates: list[float] taux zéro continus (decimal, ex 0.03)
    return: fonction DF(t) interpolation linéaire des taux zéro
    """
    tenors = np.array(tenors, dtype=float)
    rates = np.array(zero_rates, dtype=float)

    def DF(t):
        if t <= 0:
            return 1.0
        r = np.interp(t, tenors, rates)
        return np.exp(-r * t)

    return DF


def bond_schedule(maturity, freq):
    """
    Génère les dates de coupon en années depuis t=0.
    freq: nombre de coupons par an (1, 2, 4)
    """
    n = int(round(maturity * freq))
    dt = 1.0 / freq
    return [(i + 1) * dt for i in range(n)]


def bond_dirty_price(coupon_rate, maturity, ytm, freq, notional=100.0):
    """
    Prix dirty d'un bond à coupon fixe, YTM composé freq fois/an.
    coupon_rate, ytm: décimal annualisé
    """
    dates = bond_schedule(maturity, freq)
    c = coupon_rate * notional / freq
    price = 0.0
    for t in dates:
        price += c / (1 + ytm / freq) ** (t * freq)
    price += notional / (1 + ytm / freq) ** (maturity * freq)
    return price


def forward_rate(DF, t1, t2):
    """Forward simple entre t1 et t2, base ACT (year fraction = t2 - t1)."""
    tau = t2 - t1
    return (DF(t1) / DF(t2) - 1.0) / tau


def asw_spread(coupon_rate, maturity, dirty_price, fix_freq, flt_freq, DF,
               notional=100.0):
    """
    Calcule le par asset swap spread (en décimal).
    fix_freq: fréquence jambe fixe = fréquence coupons bond
    flt_freq: fréquence jambe flottante (ex 4 pour 3M Euribor)
    """
    # Jambe fixe = coupons du bond
    fix_dates = bond_schedule(maturity, fix_freq)
    tau_fix = 1.0 / fix_freq
    pv_fix = sum(coupon_rate * notional * tau_fix * DF(t) for t in fix_dates)

    # Jambe flottante
    flt_dates = bond_schedule(maturity, flt_freq)
    tau_flt = 1.0 / flt_freq
    pv_flt_libor = 0.0
    annuity_flt = 0.0
    t_prev = 0.0
    for t in flt_dates:
        L = forward_rate(DF, t_prev, t)
        pv_flt_libor += L * notional * tau_flt * DF(t)
        annuity_flt += tau_flt * DF(t)
        t_prev = t

    upfront = dirty_price - notional  # soulte
    spread = (pv_fix - pv_flt_libor - upfront) / (annuity_flt * notional)
    return spread, {
        "pv_fix": pv_fix,
        "pv_flt_libor": pv_flt_libor,
        "annuity_flt": annuity_flt,
        "upfront": upfront,
    }


def cashflow_table(coupon_rate, maturity, fix_freq, flt_freq, spread, DF,
                   notional=100.0):
    """Construit un DataFrame des cash flows pour affichage."""
    import pandas as pd

    rows = []
    fix_dates = set(np.round(bond_schedule(maturity, fix_freq), 6))
    flt_dates = bond_schedule(maturity, flt_freq)
    tau_fix = 1.0 / fix_freq
    tau_flt = 1.0 / flt_freq

    t_prev = 0.0
    for t in flt_dates:
        L = forward_rate(DF, t_prev, t)
        cf_flt = -(L + spread) * notional * tau_flt
        cf_fix = 0.0
        if round(t, 6) in fix_dates:
            cf_fix = coupon_rate * notional * tau_fix
        rows.append({
            "t (années)": round(t, 4),
            "DF": round(DF(t), 6),
            "Libor fwd": round(L, 6),
            "CF fixe reçu": round(cf_fix, 4),
            "CF flottant payé": round(cf_flt, 4),
            "CF net": round(cf_fix + cf_flt, 4),
        })
        t_prev = t

    return pd.DataFrame(rows)
