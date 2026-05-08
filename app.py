from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap", layout="wide", initial_sidebar_state="collapsed")

DARK   = "plotly_dark"
MARGIN = dict(t=40, b=40, l=50, r=40)
H      = 420
CA, CB, CC = "#00b4d8", "#ef233c", "#f4a261"

# ── CONVENTIONS ───────────────────────────────────────────────────────────────
# Compounding: annual (all discount factors use annual spot rates)
# Coupons: paid at t = k/freq for k = 1, 2, ..., n_coupons, then maturity T
# Stub: SHORT STUB — if T is not at coupon date, accrued coupon from last coupon to T
# Settlement: t=0, no accrued interest adjustment

# ── MATH ──────────────────────────────────────────────────────────────────────

def discount_factors(rate: float, T: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate coupon dates and discount factors."""
    n_coupons = int(np.floor(T * freq))
    t_coupons = np.array([k / freq for k in range(1, n_coupons + 1)])
    
    if T > t_coupons[-1] + 1e-12:
        t = np.append(t_coupons, T)
    else:
        t = t_coupons
    
    t = np.unique(np.round(t, 12))
    df = np.exp(-rate * t)
    
    return t, df


def cashflows(coupon: float, T: float, F: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Build coupon and principal cash flows."""
    t, _ = discount_factors(0, T, freq)
    n_coupons = int(np.floor(T * freq))
    coupon_period = 1 / freq
    last_coupon_date = n_coupons / freq
    
    cf = np.zeros(len(t))
    
    for i in range(min(n_coupons, len(t) - 1)):
        cf[i] = coupon * coupon_period * F
    
    if abs(T - last_coupon_date) < 1e-12:
        cf[-1] = coupon * coupon_period * F + F
    else:
        stub_years = T - last_coupon_date
        stub_frac = stub_years * freq
        cf[-1] = coupon * stub_frac * coupon_period * F + F
    
    return t, cf


def price_dirty(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate dirty bond price."""
    t, cf = cashflows(coupon, T, F, freq)
    df = np.exp(-ytm * t)
    return float(np.dot(cf, df))


def annuity_factor(ytm: float, T: float, freq: int) -> float:
    """Calculate annuity factor."""
    n_coupons = int(np.floor(T * freq))
    coupon_period = 1 / freq
    t_coupons = np.array([k / freq for k in range(1, n_coupons + 1)])
    df_coupons = np.exp(-ytm * t_coupons)
    return float(np.sum(df_coupons * coupon_period))


def par_asw(coupon: float, T: float, ytm: float, rf: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate par asset swap spread."""
    P = price_dirty(coupon, T, ytm, F, freq) / F
    ann_ytm = annuity_factor(ytm, T, freq)
    ann_rf = annuity_factor(rf, T, freq)
    df_terminal = np.exp(-rf * T)
    
    if ann_rf < 1e-12:
        return float("nan")
    
    spread = (coupon * ann_ytm + df_terminal - P) / ann_rf
    return float(spread * 1e4)


def z_spread(coupon: float, T: float, ytm: float, rf: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate Z-spread."""
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
    """Calculate yield ASW."""
    return float((ytm - rf) * 1e4)


def soulte(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate soulte."""
    P = price_dirty(coupon, T, ytm, F, freq)
    return float(P - F)


def macaulay_duration(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate Macaulay duration."""
    t, cf = cashflows(coupon, T, F, freq)
    df = np.exp(-ytm * t)
    P = float(np.dot(cf, df))
    
    if P < 1e-12:
        return float("nan")
    
    D_mac = float(np.dot(t * cf, df)) / P
    return D_mac


def modified_duration(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate modified duration."""
    D_mac = macaulay_duration(coupon, T, ytm, F, freq)
    
    if np.isnan(D_mac):
        return float("nan")
    
    return float(D_mac / (1 + ytm / freq))


def dv01(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate DV01."""
    P = price_dirty(coupon, T, ytm, F, freq)
    MD = modified_duration(coupon, T, ytm, F, freq)
    
    if np.isnan(MD):
        return float("nan")
    
    return float(MD * P * 0.0001)


# ── STREAMLIT INTERFACE ───────────────────────────────────────────────────────

st.markdown("""
    <style>
    h1 { font-size: 28px; margin-bottom: 15px; }
    [data-testid="metric-container"] {
        background-color: rgba(28, 35, 45, 0.8);
        padding: 12px 15px;
        border-radius: 8px;
        border-left: 3px solid #00b4d8;
    }
    [data-testid="metric-container"] > div:first-child { font-size: 11px; color: #888; }
    [data-testid="metric-container"] > div:last-child { font-size: 18px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("Asset Swap Pricer")

# ── INPUT SLIDERS ─────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5, col6 = st.columns(6, gap="small")

with col1:
    face = st.slider("Face", 50.0, 500.0, 100.0, 1.0, key="face_slider", format="%.1f")

with col2:
    coupon_pct = st.slider("Coupon (%)", 0.5, 15.0, 6.5, 0.05, key="coupon_slider", format="%.2f")
    coupon = coupon_pct / 100

with col3:
    mat = st.slider("Maturity (yr)", 0.5, 30.0, 7.0, 0.25, key="mat_slider", format="%.2f")

with col4:
    freq_label = st.selectbox("Freq", ["Annual", "Semi", "Quarterly"], index=1, key="freq_select")
    freq_map = {"Annual": 1, "Semi": 2, "Quarterly": 4}
    freq = freq_map[freq_label]

with col5:
    ytm_pct = st.slider("YTM (%)", 0.5, 20.0, 8.0, 0.05, key="ytm_slider", format="%.2f")
    ytm = ytm_pct / 100

with col6:
    rf_pct = st.slider("Risk-Free (%)", 0.1, 15.0, 3.0, 0.05, key="rf_slider", format="%.2f")
    rf = rf_pct / 100

st.divider()

# ── CALCULATIONS & DISPLAY ────────────────────────────────────────────────────

try:
    P = price_dirty(coupon, mat, ytm, face, freq) / face * 100
    slt = soulte(coupon, mat, ytm, face, freq)
    par_asw_val = par_asw(coupon, mat, ytm, rf, face, freq)
    z_sp = z_spread(coupon, mat, ytm, rf, face, freq)
    y_asw = yield_asw(coupon, mat, ytm, rf, face, freq)
    md = modified_duration(coupon, mat, ytm, face, freq)
    dv01_val = dv01(coupon, mat, ytm, face, freq)
    ann = annuity_factor(ytm, mat, freq)
    
    # ── METRICS ROW ───────────────────────────────────────────────────────────
    
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8, gap="small")
    
    with m1:
        st.metric("Dirty Price", f"{P:.4f}")
    with m2:
        st.metric("Soulte", f"{slt:.4f}")
    with m3:
        st.metric("Par ASW", f"{par_asw_val:.2f} bps")
    with m4:
        st.metric("Z-Spread", f"{z_sp:.2f} bps")
    with m5:
        st.metric("Yield ASW", f"{y_asw:.2f} bps")
    with m6:
        st.metric("Mod. Duration", f"{md:.4f} yr")
    with m7:
        st.metric("DV01", f"{dv01_val:.6f}")
    with m8:
        st.metric("Annuity", f"{ann:.6f}")
    
    st.divider()
    
    # ── GRAPH GENERATION ──────────────────────────────────────────────────────
    
    ytm_grid = np.linspace(max(0.002, ytm - 0.08), min(0.25, ytm + 0.12), 120)
    mat_grid = np.linspace(0.5, 30.0, 60)
    
    # Pre-calculate common arrays
    prices = [price_dirty(coupon, mat, y, face, freq) / face * 100 for y in ytm_grid]
    par_asws = [par_asw(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    z_spreads = [z_spread(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    y_asws = [yield_asw(coupon, mat, y, rf, face, freq) for y in ytm_grid]
    soultes = [soulte(coupon, mat, y, face, freq) for y in ytm_grid]
    durations = [modified_duration(coupon, mat, y, face, freq) for y in ytm_grid]
    dv01s = [dv01(coupon, mat, y, face, freq) for y in ytm_grid]
    
    # ── FIGURE 1: Bond Price vs YTM ───────────────────────────────────────────
    
    fig1 = go.Figure()
    fig1.add_scatter(x=ytm_grid*100, y=prices, mode="lines", name="Bond Price",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.2f}<extra></extra>")
    fig1.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig1.update_layout(template=DARK, height=H, margin=MARGIN, title="Bond Price vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Price (%)", hovermode="x unified", showlegend=False)
    
    # ── FIGURE 2: Par ASW vs YTM ──────────────────────────────────────────────
    
    fig2 = go.Figure()
    fig2.add_scatter(x=ytm_grid*100, y=par_asws, mode="lines", name="Par ASW",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.1f} bps<extra></extra>")
    fig2.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig2.update_layout(template=DARK, height=H, margin=MARGIN, title="Par ASW vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Par ASW (bps)", hovermode="x unified", showlegend=False)
    
    # ── FIGURE 3: Soulte vs YTM ───────────────────────────────────────────────
    
    fig3 = go.Figure()
    fig3.add_scatter(x=ytm_grid*100, y=soultes, mode="lines", name="Soulte",
                     line=dict(color=CC, width=2.5), hovertemplate="%{x:.2f}% → %{y:.4f}<extra></extra>")
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, annotation_text="At Par", annotation_position="right")
    fig3.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig3.update_layout(template=DARK, height=H, margin=MARGIN, title="Soulte vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Soulte (price)", hovermode="x unified", showlegend=False)
    
    # ── FIGURE 4: Z-Spread vs YTM ─────────────────────────────────────────────
    
    fig4 = go.Figure()
    fig4.add_scatter(x=ytm_grid*100, y=z_spreads, mode="lines", name="Z-Spread",
                     line=dict(color=CB, width=2.5), hovertemplate="%{x:.2f}% → %{y:.1f} bps<extra></extra>")
    fig4.add_vline(x=ytm_pct, line_dash="dash", line_color=CA, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig4.update_layout(template=DARK, height=H, margin=MARGIN, title="Z-Spread vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Z-Spread (bps)", hovermode="x unified", showlegend=False)
    
    # ── FIGURE 5: Modified Duration vs YTM ────────────────────────────────────
    
    fig5 = go.Figure()
    fig5.add_scatter(x=ytm_grid*100, y=durations, mode="lines", name="Mod. Duration",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.4f} yr<extra></extra>")
    fig5.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig5.update_layout(template=DARK, height=H, margin=MARGIN, title="Modified Duration vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Duration (yr)", hovermode="x unified", showlegend=False)
    
    # ── FIGURE 6: DV01 vs YTM ─────────────────────────────────────────────────
    
    fig6 = go.Figure()
    fig6.add_scatter(x=ytm_grid*100, y=dv01s, mode="lines", name="DV01",
                     line=dict(color=CC, width=2.5), hovertemplate="%{x:.2f}% → %{y:.6f}<extra></extra>")
    fig6.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig6.update_layout(template=DARK, height=H, margin=MARGIN, title="DV01 vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="DV01", hovermode="x unified", showlegend=False)
    
    # ── FIGURE 7: Par ASW Surface (Maturity × YTM) ────────────────────────────
    
    @st.cache_data(show_spinner=False, ttl=3600)
    def _surface(c, r_, F, freq_):
        Z = np.array([
            [par_asw(c, m_, y_, r_, F, freq_) for y_ in ytm_grid]
            for m_ in mat_grid
        ])
        return mat_grid, ytm_grid, Z
    
    mats_s, ytms_s, Z_s = _surface(coupon, rf, face, freq)
    
    fig7 = go.Figure(go.Surface(
        x=ytms_s*100, y=mats_s, z=Z_s,
        colorscale="Viridis",
        colorbar=dict(title="Par ASW<br>(bps)", thickness=12, len=0.7, tickfont=dict(size=9))
    ))
    fig7.update_layout(
        scene=dict(
            xaxis=dict(title="YTM (%)", tickfont=dict(size=9)),
            yaxis=dict(title="Maturity (yr)", tickfont=dict(size=9)),
            zaxis=dict(title="Par ASW (bps)", tickfont=dict(size=9)),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        title="Par ASW Surface — Maturity × YTM",
        template=DARK,
        height=H + 50,
        margin=dict(t=40, b=10, l=10, r=10)
    )
    
    # ── LAYOUT: 3 COLUMNS ─────────────────────────────────────────────────────
    
    st.markdown("#### 1. Fundamentals")
    col_a, col_b, col_c = st.columns(3, gap="medium")
    with col_a:
        st.plotly_chart(fig1, use_container_width=True, key="fig1")
    with col_b:
        st.plotly_chart(fig2, use_container_width=True, key="fig2")
    with col_c:
        st.plotly_chart(fig3, use_container_width=True, key="fig3")
    
    st.markdown("#### 2. Risk Metrics")
    col_d, col_e, col_f = st.columns(3, gap="medium")
    with col_d:
        st.plotly_chart(fig4, use_container_width=True, key="fig4")
    with col_e:
        st.plotly_chart(fig5, use_container_width=True, key="fig5")
    with col_f:
        st.plotly_chart(fig6, use_container_width=True, key="fig6")
    
    st.markdown("#### 3. Multi-Factor Surface")
    st.plotly_chart(fig7, use_container_width=True, key="fig7")

except Exception as e:
    st.error(f"**Calculation Error:** {str(e)}")
    import traceback
    st.text(traceback.format_exc())
