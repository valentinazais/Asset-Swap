from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap", layout="wide", initial_sidebar_state="collapsed")

DARK   = "plotly_dark"
MARGIN = dict(t=50, b=50, l=55, r=55)
H      = 370
CA, CB, CC = "#00b4d8", "#ef233c", "#f4a261"

# ── CONVENTIONS ───────────────────────────────────────────────────────────────
# Compounding: annual (all discount factors use annual spot rates)
# Coupons: paid at t = k/freq for k = 1, 2, ..., n_coupons, then maturity T
# Stub: SHORT STUB — if T is not at coupon date, accrued coupon from last coupon to T
# Settlement: t=0, no accrued interest adjustment
# Rates: input as % (e.g., 5.5), stored internally as decimal (0.055)

# ── MATH ──────────────────────────────────────────────────────────────────────

def discount_factors(rate: float, T: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate cash flow dates and corresponding discount factors.
    
    Args:
        rate: annual spot rate (decimal, e.g., 0.05 for 5%)
        T: maturity in years
        freq: coupon frequency (1=annual, 2=semi, 4=quarterly)
    
    Returns:
        t: array of all cash flow dates [1/freq, 2/freq, ..., T]
        df: discount factors [exp(-rate * t_i)] using continuous compounding
    """
    # All coupon dates k/freq where k = 1, 2, ..., n_coupons
    n_coupons = int(np.floor(T * freq))
    t_coupons = np.array([k / freq for k in range(1, n_coupons + 1)])
    
    # Always include maturity date T
    if T > t_coupons[-1] + 1e-12:
        t = np.append(t_coupons, T)
    else:
        t = t_coupons
    
    # Remove duplicates at machine precision
    t = np.unique(np.round(t, 12))
    df = np.exp(-rate * t)
    
    return t, df


def cashflows(coupon: float, T: float, F: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build coupon and principal cash flows.
    
    Coupon dates: k/freq for k = 1, 2, ..., n_coupons
    - If T == n_coupons/freq: last coupon is full, no stub
    - If T > n_coupons/freq: last coupon is accrued from n_coupons/freq to T
    
    Args:
        coupon: annual coupon rate (decimal)
        T: maturity in years
        F: face value
        freq: coupon frequency
    
    Returns:
        t: cash flow dates
        cf: cash flow amounts
    """
    t, _ = discount_factors(0, T, freq)
    n_coupons = int(np.floor(T * freq))
    coupon_period = 1 / freq
    last_coupon_date = n_coupons / freq
    
    cf = np.zeros(len(t))
    
    # Regular full coupons at coupon dates (not including terminal date)
    for i in range(min(n_coupons, len(t) - 1)):
        cf[i] = coupon * coupon_period * F
    
    # Terminal cash flow: coupon (full or stub) + principal at maturity T
    if abs(T - last_coupon_date) < 1e-12:
        # T is exactly at a coupon date → full final coupon
        cf[-1] = coupon * coupon_period * F + F
    else:
        # T is between coupon dates (short stub) → accrued coupon
        stub_years = T - last_coupon_date
        stub_frac = stub_years * freq
        cf[-1] = coupon * stub_frac * coupon_period * F + F
    
    return t, cf


def price_dirty(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate dirty bond price (including accrued interest).
    
    Args:
        coupon: annual coupon rate (decimal)
        T: time to maturity in years
        ytm: yield to maturity (decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        Dirty price as % of face value
    """
    t, cf = cashflows(coupon, T, F, freq)
    df = np.exp(-ytm * t)
    return float(np.dot(cf, df))


def annuity_factor(ytm: float, T: float, freq: int) -> float:
    """
    Calculate annuity factor: sum of discount factors at coupon dates (not maturity).
    
    Used for par ASW calculations.
    
    Args:
        ytm: yield (decimal)
        T: maturity (years)
        freq: coupon frequency
    
    Returns:
        Annuity factor (dimensionless)
    """
    n_coupons = int(np.floor(T * freq))
    coupon_period = 1 / freq
    
    # Discount factors at coupon dates only (k/freq for k=1..n_coupons)
    t_coupons = np.array([k / freq for k in range(1, n_coupons + 1)])
    df_coupons = np.exp(-ytm * t_coupons)
    
    # Sum of discounted period lengths
    return float(np.sum(df_coupons * coupon_period))


def par_asw(coupon: float, T: float, ytm: float, rf: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate par asset swap spread (parallel shift to risk-free curve).
    
    Par ASW is the constant spread s such that swapping bond coupons for
    risk-free + s produces zero mark-to-market.
    
    Formula:
        s = (coupon * Ann(ytm) + DF_T(rf) - P(ytm)) / Ann(rf) * 10000 bps
    
    Args:
        coupon: annual coupon (decimal)
        T: maturity (years)
        ytm: bond yield (decimal)
        rf: risk-free rate (decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        Par ASW in basis points
    """
    P = price_dirty(coupon, T, ytm, F, freq) / F
    
    ann_ytm = annuity_factor(ytm, T, freq)
    ann_rf = annuity_factor(rf, T, freq)
    
    # Discount factor at maturity under risk-free curve
    t_terminal = int(np.floor(T * freq)) / freq
    if T > t_terminal + 1e-12:
        t_terminal = T
    df_terminal = np.exp(-rf * T)
    
    if ann_rf < 1e-12:
        return float("nan")
    
    spread = (coupon * ann_ytm + df_terminal - P) / ann_rf
    return float(spread * 1e4)


def z_spread(coupon: float, T: float, ytm: float, rf: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate Z-spread (constant spread over entire risk-free curve).
    
    Solves: P = sum_i CF_i * exp(-(rf + z_spread) * t_i)
    
    Args:
        coupon: annual coupon (decimal)
        T: maturity (years)
        ytm: bond yield (decimal)
        rf: risk-free rate (decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        Z-spread in basis points
    """
    P = price_dirty(coupon, T, ytm, F, freq)
    t, cf = cashflows(coupon, T, F, freq)
    
    def pv_residual(s):
        df = np.exp(-(rf + s) * t)
        return float(np.dot(cf, df)) - P
    
    try:
        z = brentq(pv_residual, -0.5, 5.0, xtol=1e-10)
        return float(z * 1e4)
    except ValueError:
        return float("nan")


def yield_asw(coupon: float, T: float, ytm: float, rf: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate yield ASW (simple difference).
    
    Returns:
        ytm - rf in basis points
    """
    return float((ytm - rf) * 1e4)


def soulte(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate soulte (cash adjustment at trade initiation).
    
    Soulte = Dirty_Price - Par (for par ASW).
    
    Args:
        coupon: annual coupon (decimal)
        T: maturity (years)
        ytm: bond yield (decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        Soulte as % of face value
    """
    P = price_dirty(coupon, T, ytm, F, freq)
    return float(P - F)


def macaulay_duration(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate Macaulay duration.
    
    D_mac = (sum_i t_i * CF_i * DF_i) / P
    
    Args:
        coupon: annual coupon (decimal)
        T: maturity (years)
        ytm: bond yield (decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        Macaulay duration in years
    """
    t, cf = cashflows(coupon, T, F, freq)
    df = np.exp(-ytm * t)
    P = float(np.dot(cf, df))
    
    if P < 1e-12:
        return float("nan")
    
    D_mac = float(np.dot(t * cf, df)) / P
    return D_mac


def modified_duration(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate modified duration (price sensitivity).
    
    D_mod = D_mac / (1 + ytm / freq)
    
    For annual compounding (freq=1): D_mod = D_mac / (1 + ytm)
    For semi-annual (freq=2): D_mod = D_mac / (1 + ytm/2)
    
    Args:
        coupon: annual coupon (decimal)
        T: maturity (years)
        ytm: bond yield (decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        Modified duration in years
    """
    D_mac = macaulay_duration(coupon, T, ytm, F, freq)
    
    if np.isnan(D_mac):
        return float("nan")
    
    return float(D_mac / (1 + ytm / freq))


def dv01(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Calculate DV01 (dollar value of 1 basis point move).
    
    DV01 = Mod_Duration * Price * 0.0001
    
    Args:
        coupon: annual coupon (decimal)
        T: maturity (years)
        ytm: bond yield (decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        DV01 in currency units (per 100 of face)
    """
    P = price_dirty(coupon, T, ytm, F, freq)
    MD = modified_duration(coupon, T, ytm, F, freq)
    
    if np.isnan(MD):
        return float("nan")
    
    return float(MD * P * 0.0001)


# ── STREAMLIT INTERFACE ───────────────────────────────────────────────────────

st.markdown(
    """<style>
    h1 { font-size: 28px; margin-bottom: 20px; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
    </style>""",
    unsafe_allow_html=True
)

st.title("Asset Swap Pricer")

# ── INPUT CONTROLS ────────────────────────────────────────────────────────────

col_face, col_coupon, col_mat, col_freq, col_ytm, col_rf = st.columns(6)

with col_face:
    face = st.number_input("Face", value=100.0, min_value=1.0, step=1.0)

with col_coupon:
    coupon_pct = st.number_input("Coupon (%)", value=6.50, min_value=0.0, step=0.05)
    coupon = coupon_pct / 100

with col_mat:
    mat = st.number_input("Maturity (yr)", value=7.0, min_value=0.01, step=0.25)

with col_freq:
    freq_label = st.selectbox("Freq", ["Annual", "Semi", "Quarterly"], index=1)
    freq_map = {"Annual": 1, "Semi": 2, "Quarterly": 4}
    freq = freq_map[freq_label]

with col_ytm:
    ytm_pct = st.number_input("YTM (%)", value=8.00, min_value=0.01, step=0.05)
    ytm = ytm_pct / 100

with col_rf:
    rf_pct = st.number_input("Risk-Free (%)", value=3.00, min_value=0.01, step=0.05)
    rf = rf_pct / 100

if st.button("Reset", key="reset"):
    st.rerun()

st.divider()

# ── CALCULATIONS ──────────────────────────────────────────────────────────────

try:
    P = price_dirty(coupon, mat, ytm, face, freq) / face * 100
    slt = soulte(coupon, mat, ytm, face, freq)
    par_asw_val = par_asw(coupon, mat, ytm, rf, face, freq)
    z_sp = z_spread(coupon, mat, ytm, rf, face, freq)
    y_asw = yield_asw(coupon, mat, ytm, rf, face, freq)
    md = modified_duration(coupon, mat, ytm, face, freq)
    dv01_val = dv01(coupon, mat, ytm, face, freq)
    ann = annuity_factor(ytm, mat, freq)
    
    # Display metrics
    metric_cols = st.columns(9)
    metrics = [
        ("Dirty Price", f"{P:.4f}", None),
        ("Soulte", f"{slt:.4f}", None),
        ("Par ASW (bps)", f"{par_asw_val:.2f}", None),
        ("Z-Spread (bps)", f"{z_sp:.2f}", None),
        ("Yield ASW (bps)", f"{y_asw:.2f}", None),
        ("Mod. Duration", f"{md:.4f}", None),
        ("DV01", f"{dv01_val:.6f}", None),
        ("Annuity", f"{ann:.7f}", None),
    ]
    
    for i, (label, value, unit) in enumerate(metrics):
        with metric_cols[i]:
            st.metric(label, value)
    
    st.divider()
    
    # ── FIGURE 1: Price vs YTM ────────────────────────────────────────────────
    
    ytm_grid = np.linspace(0.002, 0.15, 100)
    prices = [price_dirty(coupon, mat, y, face, freq) / face * 100 for y in ytm_grid]
    
    fig1 = go.Figure()
    fig1.add_scatter(
        x=ytm_grid*100, y=prices, mode="lines",
        name="Bond Price", line=dict(color=CA, width=2)
    )
    fig1.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, annotation_text="YTM")
    fig1.update_layout(
        template=DARK, height=H, margin=MARGIN,
        title="Bond Price vs YTM",
        xaxis_title="YTM (%)",
        yaxis_title="Price (% face)",
        hovermode="x unified"
    )
    
    # ── FIGURE 2: Par ASW vs YTM ──────────────────────────────────────────────
    
    par_asws = [par_asw(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    
    fig2 = go.Figure()
    fig2.add_scatter(
        x=ytm_grid*100, y=par_asws, mode="lines",
        name="Par ASW", line=dict(color=CA, width=2)
    )
    fig2.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, annotation_text="YTM")
    fig2.update_layout(
        template=DARK, height=H, margin=MARGIN,
        title="Par ASW vs YTM",
        xaxis_title="YTM (%)",
        yaxis_title="Par ASW (bps)",
        hovermode="x unified"
    )
    
    # ── FIGURE 3: Z-Spread vs YTM ─────────────────────────────────────────────
    
    z_spreads = [z_spread(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    
    fig3 = go.Figure()
    fig3.add_scatter(
        x=ytm_grid*100, y=z_spreads, mode="lines",
        name="Z-Spread", line=dict(color=CC, width=2)
    )
    fig3.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, annotation_text="YTM")
    fig3.update_layout(
        template=DARK, height=H, margin=MARGIN,
        title="Z-Spread vs YTM",
        xaxis_title="YTM (%)",
        yaxis_title="Z-Spread (bps)",
        hovermode="x unified"
    )
    
    # ── FIGURE 4: Spread Conventions Comparison ────────────────────────────────
    
    par_asws_cmp = [par_asw(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    z_spreads_cmp = [z_spread(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    y_asws_cmp = [yield_asw(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    
    fig4 = go.Figure()
    fig4.add_scatter(
        x=ytm_grid*100, y=par_asws_cmp, mode="lines",
        name="Par ASW", line=dict(color=CA, width=2)
    )
    fig4.add_scatter(
        x=ytm_grid*100, y=z_spreads_cmp, mode="lines",
        name="Z-Spread", line=dict(color=CB, width=2, dash="dash")
    )
    fig4.add_scatter(
        x=ytm_grid*100, y=y_asws_cmp, mode="lines",
        name="Yield ASW", line=dict(color=CC, width=2, dash="dot")
    )
    fig4.add_vline(x=ytm_pct, line_dash="dash", line_color="gray")
    fig4.update_layout(
        template=DARK, height=H, margin=MARGIN,
        title="Spread Conventions vs YTM",
        xaxis_title="YTM (%)",
        yaxis_title="Spread (bps)",
        hovermode="x unified"
    )
    
    # ── FIGURE 5: Soulte vs YTM ───────────────────────────────────────────────
    
    soultes = [soulte(coupon, mat, y, face, freq) for y in ytm_grid]
    
    fig5 = go.Figure()
    fig5.add_scatter(
        x=ytm_grid*100, y=soultes, mode="lines",
        name="Soulte", line=dict(color=CB, width=2)
    )
    fig5.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="At Par")
    fig5.add_vline(x=ytm_pct, line_dash="dash", line_color="gray", annotation_text="YTM")
    fig5.update_layout(
        template=DARK, height=H, margin=MARGIN,
        title="Soulte vs YTM (zero at par)",
        xaxis_title="YTM (%)",
        yaxis_title="Soulte (price)",
        hovermode="x unified"
    )
    
    # ── FIGURE 6: Duration & DV01 ─────────────────────────────────────────────
    
    md_grid = [modified_duration(coupon, mat, y, face, freq) for y in ytm_grid]
    dv01_grid = [dv01(coupon, mat, y, face, freq) for y in ytm_grid]
    
    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.add_scatter(
        x=ytm_grid*100, y=md_grid, name="Mod. Duration",
        line=dict(color=CA, width=2), secondary_y=False
    )
    fig6.add_scatter(
        x=ytm_grid*100, y=dv01_grid, name="DV01",
        line=dict(color=CC, width=2, dash="dash"), secondary_y=True
    )
    fig6.add_vline(x=ytm_pct, line_dash="dash", line_color="gray")
    fig6.update_layout(
        template=DARK, height=H, margin=MARGIN,
        title="Duration & DV01 vs YTM",
        hovermode="x unified"
    )
    fig6.update_xaxes(title_text="YTM (%)")
    fig6.update_yaxes(title_text="Mod. Duration", secondary_y=False)
    fig6.update_yaxes(title_text="DV01", secondary_y=True)
    
    # ── FIGURE 7: Par ASW Surface (Maturity × YTM) ─────────────────────────────
    
    @st.cache_data(show_spinner=False)
    def _surface(c, r_, F, freq_):
        mats_ = np.linspace(0.5, 15, 40)
        ytms_ = np.linspace(max(0.002, r_ - 0.02), r_ + 0.15, 40)
        Z = np.array([
            [par_asw(c, m_, y_, r_, F, freq_) for y_ in ytms_]
            for m_ in mats_
        ])
        return mats_, ytms_, Z
    
    mats_s, ytms_s, Z_s = _surface(coupon, rf, face, freq)
    
    fig7 = go.Figure(go.Surface(
        x=ytms_s*100, y=mats_s, z=Z_s,
        colorscale="Viridis",
        colorbar=dict(title="Par ASW (bps)", thickness=15, len=0.7)
    ))
    fig7.update_layout(
        scene=dict(
            xaxis=dict(title="YTM (%)", tickfont=dict(size=10)),
            yaxis=dict(title="Maturity (yr)", tickfont=dict(size=10)),
            zaxis=dict(title="Par ASW (bps)", tickfont=dict(size=10)),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        title="Par ASW Surface — Maturity × YTM",
        template=DARK,
        height=H,
        margin=dict(t=50, b=10, l=10, r=10)
    )
    
    # ── LAYOUT: 2×4 GRID ──────────────────────────────────────────────────────
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(fig1, use_container_width=True, key="fig1")
    with col2:
        st.plotly_chart(fig2, use_container_width=True, key="fig2")
    with col3:
        st.plotly_chart(fig3, use_container_width=True, key="fig3")
    with col4:
        st.plotly_chart(fig4, use_container_width=True, key="fig4")
    
    col5, col6, col7 = st.columns(3)
    with col5:
        st.plotly_chart(fig5, use_container_width=True, key="fig5")
    with col6:
        st.plotly_chart(fig6, use_container_width=True, key="fig6")
    with col7:
        st.plotly_chart(fig7, use_container_width=True, key="fig7")

except Exception as e:
    st.error(f"**Calculation Error:** {str(e)}")
    st.stop()
