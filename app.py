# Asset Swap Pricer — run with: streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap Pricer", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────────

def bond_price(
    coupon: float,
    maturity: float,
    ytm: float,
    face: float = 100.0,
    frequency: int = 2,
) -> float:
    """Dirty price of a fixed-rate bond (flat yield curve)."""
    n = int(round(maturity * frequency))
    dt = 1.0 / frequency
    c = coupon / frequency * face
    times = np.arange(1, n + 1) * dt
    df = np.exp(-ytm * times)
    return float(np.sum(c * df) + face * df[-1])


def accrued_interest(coupon: float, face: float, frequency: int, t_since_last: float = 0.0) -> float:
    """Simple accrued interest since last coupon."""
    return coupon * face * t_since_last


def par_asset_swap_spread(
    coupon: float,
    maturity: float,
    ytm: float,
    risk_free: float,
    face: float = 100.0,
    frequency: int = 2,
) -> float:
    """
    Par asset swap spread (bps).
    Spread s such that:
      (Bond dirty price - Face) = PV of spread payments on floating leg
    i.e. s = (C_fixed - C_par) / Annuity  expressed as spread over LIBOR/risk-free.
    
    Standard par ASW spread formula:
      ASW = (P - 100) / Annuity  * frequency  (annualised, in price terms)
    then expressed in bps via yield differential.
    
    Exact: s = (coupon - ytm_equivalent) adjusted for price/par difference.
    Using standard market formula:
      s = [ coupon - (1 - P/100 * e^{-rT} discounted recovery) ] / Annuity
    
    Clean market formula (par asset swap):
      s = (coupon * Annuity + 100 * Z(T) - P) / Annuity
    where P is dirty price / 100, Z(T) = e^{-r*T}, Annuity = sum of discount factors.
    """
    n = int(round(maturity * frequency))
    dt = 1.0 / frequency
    times = np.arange(1, n + 1) * dt
    df_rf = np.exp(-risk_free * times)

    annuity = float(np.sum(df_rf)) * dt * frequency  # = sum(df * dt) * frequency → dimensionless annuity per unit notional
    # Simpler: annuity = sum of df (each payment is 1/freq of notional per year)
    annuity = float(np.sum(df_rf * dt))  # PV of 1 unit paid continuously → discrete approx

    Z_T = np.exp(-risk_free * maturity)
    P = bond_price(coupon, maturity, ytm, face, frequency) / face  # normalise to 1

    # Par ASW spread (annualised, as fraction)
    # s * Annuity = coupon * Annuity + Z_T - P
    numerator = coupon * annuity + Z_T - P
    if abs(annuity) < 1e-12:
        return 0.0
    return float(numerator / annuity * 10_000)  # bps


def yield_asset_swap_spread(
    coupon: float,
    maturity: float,
    ytm: float,
    risk_free: float,
) -> float:
    """Yield-based ASW spread = YTM - risk_free (bps)."""
    return (ytm - risk_free) * 10_000


def z_spread(
    coupon: float,
    maturity: float,
    dirty_price: float,
    risk_free: float,
    face: float = 100.0,
    frequency: int = 2,
) -> float:
    """Z-spread: constant spread over risk-free curve that reprices the bond."""
    def pv(s):
        n = int(round(maturity * frequency))
        dt = 1.0 / frequency
        c = coupon / frequency * face
        times = np.arange(1, n + 1) * dt
        df = np.exp(-(risk_free + s) * times)
        return float(np.sum(c * df) + face * df[-1]) - dirty_price

    try:
        return brentq(pv, -0.20, 2.0, xtol=1e-10) * 10_000
    except Exception:
        return float("nan")


def asset_swap_soulte(
    coupon: float,
    maturity: float,
    ytm: float,
    risk_free: float,
    face: float = 100.0,
    frequency: int = 2,
) -> dict:
    """
    Full asset swap package breakdown.
    Soulte (upfront) = Dirty Price - Par  (paid by protection buyer to dealer).
    Returns all legs and metrics.
    """
    dirty = bond_price(coupon, maturity, ytm, face, frequency)
    clean = dirty  # we ignore accrued for simplicity (settlement at coupon date)
    soulte = dirty - face  # upfront: >0 premium bond, <0 discount bond

    par_asw = par_asset_swap_spread(coupon, maturity, ytm, risk_free, face, frequency)
    yield_asw = yield_asset_swap_spread(coupon, maturity, ytm, risk_free)
    zspd = z_spread(coupon, maturity, dirty, risk_free, face, frequency)

    # Annuity (risky, here risk-free for ASW)
    n = int(round(maturity * frequency))
    dt = 1.0 / frequency
    times = np.arange(1, n + 1) * dt
    df_rf = np.exp(-risk_free * times)
    annuity = float(np.sum(df_rf * dt))

    # Fixed leg PV (bond coupons discounted at risk-free)
    fixed_leg_pv = float(np.sum(coupon * face * df_rf * dt) + face * np.exp(-risk_free * maturity))

    # Floating leg PV at par = face (by definition of floating leg at par)
    floating_leg_pv = face

    # Duration (modified, approximate)
    duration = float(np.sum(times * coupon / frequency * face * df_rf) + maturity * face * np.exp(-risk_free * maturity)) / dirty
    dv01 = dirty * duration / 10_000  # per bp

    return {
        "dirty_price": dirty,
        "clean_price": clean,
        "soulte": soulte,
        "par_asw_spread_bps": par_asw,
        "yield_asw_spread_bps": yield_asw,
        "z_spread_bps": zspd,
        "fixed_leg_pv": fixed_leg_pv,
        "floating_leg_pv": floating_leg_pv,
        "annuity": annuity,
        "modified_duration": duration,
        "dv01": dv01,
        "ytm": ytm,
        "risk_free": risk_free,
        "coupon": coupon,
        "maturity": maturity,
        "face": face,
    }


def cashflow_schedule(
    coupon: float,
    maturity: float,
    ytm: float,
    risk_free: float,
    face: float = 100.0,
    frequency: int = 2,
) -> pd.DataFrame:
    """Return period-by-period cashflow table."""
    n = int(round(maturity * frequency))
    dt = 1.0 / frequency
    rows = []
    for i in range(1, n + 1):
        t = i * dt
        df_ytm = np.exp(-ytm * t)
        df_rf = np.exp(-risk_free * t)
        c = coupon / frequency * face
        principal = face if i == n else 0.0
        rows.append({
            "Time": round(t, 4),
            "Fixed Coupon": round(c, 4),
            "Principal": round(principal, 2),
            "Disc Factor (YTM)": round(df_ytm, 6),
            "Disc Factor (RF)": round(df_rf, 6),
            "PV Coupon (YTM)": round(c * df_ytm, 4),
            "PV Coupon (RF)": round(c * df_rf, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

COLORS = ["#00b4d8", "#f77f00", "#06d6a0", "#e63946", "#a8dadc", "#457b9d"]


def plot_cashflows(cf_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=cf_df["Time"], y=cf_df["Fixed Coupon"], name="Coupon", marker_color=COLORS[0])
    fig.add_bar(x=cf_df["Time"], y=cf_df["Principal"], name="Principal", marker_color=COLORS[1])
    fig.update_layout(
        barmode="stack", title="Cashflow Schedule",
        xaxis_title="Time (years)", yaxis_title="Amount",
        template="plotly_dark", height=350,
    )
    return fig


def plot_price_vs_ytm(coupon, maturity, risk_free, face, frequency) -> go.Figure:
    ytm_grid = np.linspace(max(0.001, coupon - 0.08), coupon + 0.08, 200)
    prices = [bond_price(coupon, maturity, y, face, frequency) for y in ytm_grid]
    par_asws = [par_asset_swap_spread(coupon, maturity, y, risk_free, face, frequency) for y in ytm_grid]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ytm_grid * 100, y=prices, name="Dirty Price", line=dict(color=COLORS[0])), secondary_y=False)
    fig.add_trace(go.Scatter(x=ytm_grid * 100, y=par_asws, name="Par ASW (bps)", line=dict(color=COLORS[1], dash="dash")), secondary_y=True)
    fig.add_vline(x=risk_free * 100, line_dash="dot", line_color="gray", annotation_text="Risk-Free")
    fig.update_layout(title="Price & Par ASW vs YTM", xaxis_title="YTM (%)", template="plotly_dark", height=350)
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Par ASW (bps)", secondary_y=True)
    return fig


def plot_spread_vs_param(param_name: str, param_grid, spreads_par, spreads_z, spreads_yield) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_grid, y=spreads_par, name="Par ASW", line=dict(color=COLORS[0])))
    fig.add_trace(go.Scatter(x=param_grid, y=spreads_z, name="Z-Spread", line=dict(color=COLORS[1], dash="dash")))
    fig.add_trace(go.Scatter(x=param_grid, y=spreads_yield, name="Yield ASW", line=dict(color=COLORS[2], dash="dot")))
    fig.update_layout(
        title=f"Spreads vs {param_name}",
        xaxis_title=param_name, yaxis_title="Spread (bps)",
        template="plotly_dark", height=350,
    )
    return fig


def plot_soulte_vs_param(param_name: str, param_grid, soultes) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_grid, y=soultes, mode="lines+markers", name="Soulte", line=dict(color=COLORS[3])))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Soulte vs {param_name}",
        xaxis_title=param_name, yaxis_title="Soulte (price units)",
        template="plotly_dark", height=350,
    )
    return fig


def plot_dv01_duration(coupon, maturity, risk_free, face, frequency) -> go.Figure:
    ytm_grid = np.linspace(max(0.001, risk_free - 0.06), risk_free + 0.10, 150)
    durations = []
    dv01s = []
    for y in ytm_grid:
        r = asset_swap_soulte(coupon, maturity, y, risk_free, face, frequency)
        durations.append(r["modified_duration"])
        dv01s.append(r["dv01"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ytm_grid * 100, y=durations, name="Mod. Duration", line=dict(color=COLORS[4])), secondary_y=False)
    fig.add_trace(go.Scatter(x=ytm_grid * 100, y=dv01s, name="DV01", line=dict(color=COLORS[5], dash="dash")), secondary_y=True)
    fig.update_layout(title="Duration & DV01 vs YTM", xaxis_title="YTM (%)", template="plotly_dark", height=350)
    fig.update_yaxes(title_text="Modified Duration", secondary_y=False)
    fig.update_yaxes(title_text="DV01", secondary_y=True)
    return fig


def plot_spread_surface(coupon, risk_free, face, frequency) -> go.Figure:
    maturities = np.linspace(1, 15, 30)
    ytm_grid = np.linspace(max(0.001, risk_free - 0.04), risk_free + 0.08, 30)
    Z = np.zeros((len(maturities), len(ytm_grid)))
    for i, mat in enumerate(maturities):
        for j, y in enumerate(ytm_grid):
            Z[i, j] = par_asset_swap_spread(coupon, mat, y, risk_free, face, frequency)
    fig = go.Figure(go.Surface(
        x=ytm_grid * 100, y=maturities, z=Z,
        colorscale="Viridis", name="Par ASW"
    ))
    fig.update_layout(
        title="Par ASW Surface (Maturity × YTM)",
        scene=dict(xaxis_title="YTM (%)", yaxis_title="Maturity (yr)", zaxis_title="Par ASW (bps)"),
        template="plotly_dark", height=500,
    )
    return fig


def plot_leg_pv(coupon, maturity, ytm, risk_free, face, frequency) -> go.Figure:
    ytm_grid = np.linspace(max(0.001, risk_free - 0.06), risk_free + 0.10, 100)
    fixed_pvs, floating_pvs, soultes = [], [], []
    for y in ytm_grid:
        r = asset_swap_soulte(coupon, maturity, y, risk_free, face, frequency)
        fixed_pvs.append(r["fixed_leg_pv"])
        floating_pvs.append(r["floating_leg_pv"])
        soultes.append(r["soulte"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ytm_grid * 100, y=fixed_pvs, name="Fixed Leg PV", line=dict(color=COLORS[0])))
    fig.add_trace(go.Scatter(x=ytm_grid * 100, y=floating_pvs, name="Floating Leg PV (Par)", line=dict(color=COLORS[1], dash="dash")))
    fig.add_trace(go.Scatter(x=ytm_grid * 100, y=soultes, name="Soulte", line=dict(color=COLORS[3], dash="dot")))
    fig.add_vline(x=ytm * 100, line_dash="dot", line_color="white", annotation_text="Current YTM")
    fig.update_layout(
        title="Fixed Leg PV, Floating Leg PV & Soulte vs YTM",
        xaxis_title="YTM (%)", yaxis_title="Value",
        template="plotly_dark", height=380,
    )
    return fig


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

st.title("Asset Swap Pricer")

tab1, tab2, tab3 = st.tabs(["Pricer & Soulte", "Sensitivity Analysis", "Spread Surface"])

# ── TAB 1: PRICER ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Asset Swap — Par Structure")
    st.markdown(
        "Price a fixed-rate bond packaged as an asset swap. "
        "The soulte (upfront payment) compensates for the difference between "
        "dirty price and par. Spreads are shown across par, yield and Z-spread conventions."
    )
    st.markdown("---")

    if "pr_rc" not in st.session_state:
        st.session_state.pr_rc = 0
    pr_rc = st.session_state.pr_rc
    if st.button("↻ Reset", key="pr_reset"):
        st.session_state.pr_rc += 1
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    face      = c1.number_input("Face Value", value=100.0, step=10.0, format="%.2f", key=f"pr_face_{pr_rc}")
    coupon    = c2.slider("Coupon Rate", 0.001, 0.15, 0.05, 0.001, format="%.3f", key=f"pr_cpn_{pr_rc}")
    maturity  = c3.number_input("Maturity (years)", value=5.0, step=0.5, min_value=0.5, max_value=30.0, format="%.1f", key=f"pr_mat_{pr_rc}")
    frequency = c4.selectbox("Frequency", [1, 2, 4], index=1, format_func=lambda x: {1:"Annual",2:"Semi-annual",4:"Quarterly"}[x], key=f"pr_freq_{pr_rc}")

    c5, c6 = st.columns(2)
    ytm       = c5.slider("YTM (bond yield)", 0.001, 0.15, 0.055, 0.001, format="%.3f", key=f"pr_ytm_{pr_rc}")
    risk_free = c6.slider("Risk-Free Rate", 0.0, 0.12, 0.03, 0.001, format="%.3f", key=f"pr_rf_{pr_rc}")

    result = asset_swap_soulte(coupon, maturity, ytm, risk_free, face, frequency)

    st.markdown("---")
    st.markdown("#### Key Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dirty Price", f"{result['dirty_price']:.4f}")
    m2.metric("Soulte", f"{result['soulte']:.4f}", delta=f"{'Premium' if result['soulte']>0 else 'Discount'}")
    m3.metric("Par ASW Spread", f"{result['par_asw_spread_bps']:.2f} bps")
    m4.metric("Z-Spread", f"{result['z_spread_bps']:.2f} bps")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Yield ASW Spread", f"{result['yield_asw_spread_bps']:.2f} bps")
    m6.metric("Modified Duration", f"{result['modified_duration']:.4f}")
    m7.metric("DV01", f"{result['dv01']:.6f}")
    m8.metric("Annuity", f"{result['annuity']:.6f}")

    st.markdown("---")
    st.markdown("#### Leg Decomposition & Soulte vs YTM")
    st.plotly_chart(plot_leg_pv(coupon, maturity, ytm, risk_free, face, frequency), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Cashflow Schedule")
    cf_df = cashflow_schedule(coupon, maturity, ytm, risk_free, face, frequency)
    st.plotly_chart(plot_cashflows(cf_df), use_container_width=True)
    st.dataframe(cf_df, use_container_width=True, hide_index=True)


# ── TAB 2: SENSITIVITY ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Sensitivity Analysis")
    st.markdown("Sweep each parameter independently and observe how spreads and soulte respond.")
    st.markdown("---")

    if "sa_rc" not in st.session_state:
        st.session_state.sa_rc = 0
    sa_rc = st.session_state.sa_rc
    if st.button("↻ Reset", key="sa_reset"):
        st.session_state.sa_rc += 1
        st.rerun()

    sa_c1, sa_c2, sa_c3, sa_c4 = st.columns(4)
    sa_face      = sa_c1.number_input("Face Value", value=100.0, step=10.0, format="%.2f", key=f"sa_face_{sa_rc}")
    sa_coupon    = sa_c2.slider("Coupon Rate", 0.001, 0.15, 0.05, 0.001, format="%.3f", key=f"sa_cpn_{sa_rc}")
    sa_maturity  = sa_c3.number_input("Maturity (years)", value=5.0, step=0.5, min_value=0.5, max_value=30.0, format="%.1f", key=f"sa_mat_{sa_rc}")
    sa_frequency = sa_c4.selectbox("Frequency", [1, 2, 4], index=1, format_func=lambda x: {1:"Annual",2:"Semi-annual",4:"Quarterly"}[x], key=f"sa_freq_{sa_rc}")

    sa_c5, sa_c6 = st.columns(2)
    sa_ytm       = sa_c5.slider("Base YTM", 0.001, 0.15, 0.055, 0.001, format="%.3f", key=f"sa_ytm_{sa_rc}")
    sa_rf        = sa_c6.slider("Base Risk-Free Rate", 0.0, 0.12, 0.03, 0.001, format="%.3f", key=f"sa_rf_{sa_rc}")

    st.markdown("---")
    st.markdown("#### 1 — Spreads & Soulte vs YTM")

    ytm_grid = np.linspace(max(0.001, sa_coupon - 0.08), sa_coupon + 0.08, 200)
    par_s, z_s, y_s, sol_ytm = [], [], [], []
    for y in ytm_grid:
        r = asset_swap_soulte(sa_coupon, sa_maturity, y, sa_rf, sa_face, sa_frequency)
        par_s.append(r["par_asw_spread_bps"])
        z_s.append(r["z_spread_bps"])
        y_s.append(r["yield_asw_spread_bps"])
        sol_ytm.append(r["soulte"])

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(plot_spread_vs_param("YTM (%)", ytm_grid * 100, par_s, z_s, y_s), use_container_width=True)
    col_b.plotly_chart(plot_soulte_vs_param("YTM (%)", ytm_grid * 100, sol_ytm), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 2 — Spreads & Soulte vs Risk-Free Rate")

    rf_grid = np.linspace(0.001, 0.12, 200)
    par_s2, z_s2, y_s2, sol_rf = [], [], [], []
    for r_ in rf_grid:
        r = asset_swap_soulte(sa_coupon, sa_maturity, sa_ytm, r_, sa_face, sa_frequency)
        par_s2.append(r["par_asw_spread_bps"])
        z_s2.append(r["z_spread_bps"])
        y_s2.append(r["yield_asw_spread_bps"])
        sol_rf.append(r["soulte"])

    col_c, col_d = st.columns(2)
    col_c.plotly_chart(plot_spread_vs_param("Risk-Free Rate (%)", rf_grid * 100, par_s2, z_s2, y_s2), use_container_width=True)
    col_d.plotly_chart(plot_soulte_vs_param("Risk-Free Rate (%)", rf_grid * 100, sol_rf), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3 — Spreads & Soulte vs Coupon Rate")

    cpn_grid = np.linspace(0.005, 0.15, 200)
    par_s3, z_s3, y_s3, sol_cpn = [], [], [], []
    for c_ in cpn_grid:
        r = asset_swap_soulte(c_, sa_maturity, sa_ytm, sa_rf, sa_face, sa_frequency)
        par_s3.append(r["par_asw_spread_bps"])
        z_s3.append(r["z_spread_bps"])
        y_s3.append(r["yield_asw_spread_bps"])
        sol_cpn.append(r["soulte"])

    col_e, col_f = st.columns(2)
    col_e.plotly_chart(plot_spread_vs_param("Coupon Rate (%)", cpn_grid * 100, par_s3, z_s3, y_s3), use_container_width=True)
    col_f.plotly_chart(plot_soulte_vs_param("Coupon Rate (%)", cpn_grid * 100, sol_cpn), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 4 — Spreads & Soulte vs Maturity")

    mat_grid = np.linspace(0.5, 20, 100)
    par_s4, z_s4, y_s4, sol_mat = [], [], [], []
    for m_ in mat_grid:
        r = asset_swap_soulte(sa_coupon, m_, sa_ytm, sa_rf, sa_face, sa_frequency)
        par_s4.append(r["par_asw_spread_bps"])
        z_s4.append(r["z_spread_bps"])
        y_s4.append(r["yield_asw_spread_bps"])
        sol_mat.append(r["soulte"])

    col_g, col_h = st.columns(2)
    col_g.plotly_chart(plot_spread_vs_param("Maturity (years)", mat_grid, par_s4, z_s4, y_s4), use_container_width=True)
    col_h.plotly_chart(plot_soulte_vs_param("Maturity (years)", mat_grid, sol_mat), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 5 — Duration & DV01 vs YTM")
    st.plotly_chart(plot_dv01_duration(sa_coupon, sa_maturity, sa_rf, sa_face, sa_frequency), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 6 — Price vs YTM")
    st.plotly_chart(plot_price_vs_ytm(sa_coupon, sa_maturity, sa_rf, sa_face, sa_frequency), use_container_width=True)


# ── TAB 3: SURFACE ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Par ASW Spread Surface")
    st.markdown("3D surface of Par ASW spread as a joint function of maturity and YTM.")
    st.markdown("---")

    if "sf_rc" not in st.session_state:
        st.session_state.sf_rc = 0
    sf_rc = st.session_state.sf_rc
    if st.button("↻ Reset", key="sf_reset"):
        st.session_state.sf_rc += 1
        st.rerun()

    sf_c1, sf_c2, sf_c3, sf_c4 = st.columns(4)
    sf_coupon    = sf_c1.slider("Coupon Rate", 0.01, 0.12, 0.05, 0.005, format="%.3f", key=f"sf_cpn_{sf_rc}")
    sf_rf        = sf_c2.slider("Risk-Free Rate", 0.001, 0.10, 0.03, 0.001, format="%.3f", key=f"sf_rf_{sf_rc}")
    sf_face      = sf_c3.number_input("Face Value", value=100.0, step=10.0, format="%.2f", key=f"sf_face_{sf_rc}")
    sf_frequency = sf_c4.selectbox("Frequency", [1, 2, 4], index=1, format_func=lambda x: {1:"Annual",2:"Semi-annual",4:"Quarterly"}[x], key=f"sf_freq_{sf_rc}")

    st.plotly_chart(plot_spread_surface(sf_coupon, sf_rf, sf_face, sf_frequency), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Soulte Surface (Maturity × YTM)")

    @st.cache_data(show_spinner="Computing soulte surface…")
    def _soulte_surface(coupon, rf, face, frequency):
        mats = np.linspace(1, 15, 30)
        ytms = np.linspace(max(0.001, rf - 0.04), rf + 0.08, 30)
        Z = np.zeros((len(mats), len(ytms)))
        for i, m in enumerate(mats):
            for j, y in enumerate(ytms):
                Z[i, j] = asset_swap_soulte(coupon, m, y, rf, face, frequency)["soulte"]
        return mats, ytms, Z

    mats, ytms, Z_sol = _soulte_surface(sf_coupon, sf_rf, sf_face, sf_frequency)
    fig_sol = go.Figure(go.Surface(x=ytms * 100, y=mats, z=Z_sol, colorscale="RdBu", name="Soulte"))
    fig_sol.update_layout(
        title="Soulte Surface (Maturity × YTM)",
        scene=dict(xaxis_title="YTM (%)", yaxis_title="Maturity (yr)", zaxis_title="Soulte"),
        template="plotly_dark", height=500,
    )
    st.plotly_chart(fig_sol, use_container_width=True)
