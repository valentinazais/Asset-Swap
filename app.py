from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap", layout="wide", initial_sidebar_state="collapsed")

# ── HIDE SLIDER DOTS ──────────────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stSlider"] div[role="slider"] { display: none !important; }
    [data-testid="stSlider"] .st-emotion-cache-1siy2j7 { display: none !important; }
    </style>
""", unsafe_allow_html=True)

DARK   = "plotly_dark"
MARGIN = dict(t=40, b=40, l=50, r=40)
H      = 420
CA, CB, CC = "#00b4d8", "#ef233c", "#f4a261"

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
        return 0.0
    
    spread = (P - 1) / (ann_rf + df_terminal - 1) * 10000
    return float(spread)


def z_spread(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate Z-spread (approximation)."""
    return (ytm - 0.02) * 10000


def modified_duration(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate modified duration using numerical differentiation."""
    dy = 0.0001
    P0 = price_dirty(coupon, T, ytm, F, freq)
    P_plus = price_dirty(coupon, T, ytm + dy, F, freq)
    P_minus = price_dirty(coupon, T, ytm - dy, F, freq)
    duration = -(P_plus - P_minus) / (2 * P0 * dy)
    return float(duration)


def dv01(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate DV01 (dollar value of 1 bp)."""
    P0 = price_dirty(coupon, T, ytm, F, freq)
    P_plus = price_dirty(coupon, T, ytm + 0.0001, F, freq)
    return float(abs(P_plus - P0))


# ── SESSION STATE ─────────────────────────────────────────────────────────────

if "face" not in st.session_state:
    st.session_state.face = 100.0
if "coupon" not in st.session_state:
    st.session_state.coupon = 6.75
if "maturity" not in st.session_state:
    st.session_state.maturity = 8.0
if "freq" not in st.session_state:
    st.session_state.freq = "Semi"
if "ytm" not in st.session_state:
    st.session_state.ytm = 8.0
if "rf" not in st.session_state:
    st.session_state.rf = 3.0

FREQ_MAP = {"Semi": 2, "Annual": 1, "Quarterly": 4}
freq = FREQ_MAP[st.session_state.freq]

# ── CONTROLS ──────────────────────────────────────────────────────────────────

st.markdown("## Asset Swap Pricer")

col1, col2, col3, col4, col5, col6 = st.columns(6)

# Face
with col1:
    st.markdown("**Face**")
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("−", key="btn_face_minus", use_container_width=True):
            st.session_state.face = max(10, st.session_state.face - 5)
    with bc2:
        if st.button("+", key="btn_face_plus", use_container_width=True):
            st.session_state.face = min(1000, st.session_state.face + 5)
    st.session_state.face = st.slider("Face Value", 10.0, 1000.0, st.session_state.face, 1.0, key="sl_face")
    st.markdown(f"<div style='text-align:center; color:#ef233c; font-size:18px; font-weight:bold'>{st.session_state.face:.2f}</div>", unsafe_allow_html=True)

# Coupon
with col2:
    st.markdown("**Coupon (%)**")
    bc3, bc4 = st.columns(2)
    with bc3:
        if st.button("−", key="btn_coupon_minus", use_container_width=True):
            st.session_state.coupon = max(0.01, st.session_state.coupon - 0.25)
    with bc4:
        if st.button("+", key="btn_coupon_plus", use_container_width=True):
            st.session_state.coupon = min(15, st.session_state.coupon + 0.25)
    st.session_state.coupon = st.slider("Coupon", 0.01, 15.0, st.session_state.coupon, 0.01, key="sl_coupon")
    st.markdown(f"<div style='text-align:center; color:#ef233c; font-size:18px; font-weight:bold'>{st.session_state.coupon:.2f}</div>", unsafe_allow_html=True)

# Maturity
with col3:
    st.markdown("**Maturity (yr)**")
    bc5, bc6 = st.columns(2)
    with bc5:
        if st.button("−", key="btn_maturity_minus", use_container_width=True):
            st.session_state.maturity = max(0.25, st.session_state.maturity - 0.5)
    with bc6:
        if st.button("+", key="btn_maturity_plus", use_container_width=True):
            st.session_state.maturity = min(30, st.session_state.maturity + 0.5)
    st.session_state.maturity = st.slider("Maturity", 0.25, 30.0, st.session_state.maturity, 0.25, key="sl_maturity")
    st.markdown(f"<div style='text-align:center; color:#ef233c; font-size:18px; font-weight:bold'>{st.session_state.maturity:.2f}</div>", unsafe_allow_html=True)

# Frequency
with col4:
    st.markdown("**Freq**")
    st.markdown("")
    st.markdown("")
    st.session_state.freq = st.selectbox("", ["Semi", "Annual", "Quarterly"], index=["Semi", "Annual", "Quarterly"].index(st.session_state.freq), key="sl_freq")

# YTM
with col5:
    st.markdown("**YTM (%)**")
    bc7, bc8 = st.columns(2)
    with bc7:
        if st.button("−", key="btn_ytm_minus", use_container_width=True):
            st.session_state.ytm = max(0.01, st.session_state.ytm - 0.25)
    with bc8:
        if st.button("+", key="btn_ytm_plus", use_container_width=True):
            st.session_state.ytm = min(20, st.session_state.ytm + 0.25)
    st.session_state.ytm = st.slider("YTM", 0.01, 20.0, st.session_state.ytm, 0.01, key="sl_ytm")
    st.markdown(f"<div style='text-align:center; color:#ef233c; font-size:18px; font-weight:bold'>{st.session_state.ytm:.2f}</div>", unsafe_allow_html=True)

# Risk-Free Rate
with col6:
    st.markdown("**Risk-Free (%)**")
    bc9, bc10 = st.columns(2)
    with bc9:
        if st.button("−", key="btn_rf_minus", use_container_width=True):
            st.session_state.rf = max(0.01, st.session_state.rf - 0.25)
    with bc10:
        if st.button("+", key="btn_rf_plus", use_container_width=True):
            st.session_state.rf = min(20, st.session_state.rf + 0.25)
    st.session_state.rf = st.slider("Risk-Free Rate", 0.01, 20.0, st.session_state.rf, 0.01, key="sl_rf")
    st.markdown(f"<div style='text-align:center; color:#ef233c; font-size:18px; font-weight:bold'>{st.session_state.rf:.2f}</div>", unsafe_allow_html=True)

st.markdown("---")

# ── CALCULATIONS ──────────────────────────────────────────────────────────────

try:
    face = st.session_state.face
    coupon = st.session_state.coupon / 100
    maturity = st.session_state.maturity
    ytm = st.session_state.ytm / 100
    rf = st.session_state.rf / 100
    
    dirty_price = price_dirty(coupon, maturity, ytm, face, freq)
    soulte = dirty_price - face
    par_asw_val = par_asw(coupon, maturity, ytm, rf, face, freq)
    z_spread_val = z_spread(coupon, maturity, ytm, face, freq)
    yield_asw = ytm * 10000 + par_asw_val
    mod_duration = modified_duration(coupon, maturity, ytm, face, freq)
    dv01_val = dv01(coupon, maturity, ytm, face, freq)
    ann_val = annuity_factor(ytm, maturity, freq)
    
    # ── METRICS DISPLAY ───────────────────────────────────────────────────────
    
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6, col_m7, col_m8 = st.columns(8)
    
    with col_m1:
        st.metric("Dirty Price", f"{dirty_price:.4f}")
    with col_m2:
        st.metric("Soulte", f"{soulte:.4f}")
    with col_m3:
        st.metric("Par ASW", f"{par_asw_val:.2f} bps")
    with col_m4:
        st.metric("Z-Spread", f"{z_spread_val:.2f} bps")
    with col_m5:
        st.metric("Yield ASW", f"{yield_asw:.2f} bps")
    with col_m6:
        st.metric("Mod. Duration", f"{mod_duration:.4f} yr")
    with col_m7:
        st.metric("DV01", f"{dv01_val:.6f}")
    with col_m8:
        st.metric("Annuity", f"{ann_val:.6f}")
    
    st.markdown("---")
    
    # ── GRID RANGES ───────────────────────────────────────────────────────────
    
    ytm_grid = np.linspace(0.001, 0.20, 150)
    mat_grid = np.linspace(0.25, 30, 80)
    
    # ── CACHED COMPUTATIONS ───────────────────────────────────────────────────
    
    @st.cache_data(show_spinner=False, ttl=3600)
    def _compute_curves(c, T, F, freq_):
        prices = np.array([price_dirty(c, T, y_, F, freq_) for y_ in ytm_grid])
        par_asws = np.array([par_asw(c, T, y_, rf, F, freq_) for y_ in ytm_grid])
        soultes = prices - F
        z_spreads = (ytm_grid - 0.02) * 10000
        durations = np.array([modified_duration(c, T, y_, F, freq_) for y_ in ytm_grid])
        dv01s = np.array([dv01(c, T, y_, F, freq_) for y_ in ytm_grid])
        
        return prices, par_asws, soultes, z_spreads, durations, dv01s
    
    prices, par_asws, soultes, z_spreads, durations, dv01s = _compute_curves(coupon, maturity, face, freq)
    
    # ── FIGURE 1: Bond Price vs YTM ───────────────────────────────────────────
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ytm_grid*100, y=prices, mode="lines", name="Bond Price",
        line=dict(color=CA, width=2.5),
        hovertemplate="%{x:.2f}% → %{y:.4f}<extra></extra>"
    ))
    fig1.add_vline(x=ytm*100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig1.update_layout(
        title="Bond Price vs YTM",
        xaxis_title="YTM (%)", yaxis_title="Price",
        template=DARK, height=H, margin=MARGIN,
        hovermode="x unified", showlegend=False
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 2: Par ASW vs YTM ──────────────────────────────────────────────
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ytm_grid*100, y=par_asws, mode="lines", name="Par ASW",
        line=dict(color=CA, width=2.5),
        hovertemplate="%{x:.2f}% → %{y:.2f} bps<extra></extra>"
    ))
    fig2.add_vline(x=ytm*100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig2.update_layout(
        title="Par ASW vs YTM",
        xaxis_title="YTM (%)", yaxis_title="Par ASW (bps)",
        template=DARK, height=H, margin=MARGIN,
        hovermode="x unified", showlegend=False
    )
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 3: Soulte vs YTM ───────────────────────────────────────────────
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=ytm_grid*100, y=soultes, mode="lines", name="Soulte",
        line=dict(color=CC, width=2.5),
        hovertemplate="%{x:.2f}% → %{y:.4f}<extra></extra>"
    ))
    fig3.add_vline(x=ytm*100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig3.update_layout(
        title="Soulte vs YTM",
        xaxis_title="YTM (%)", yaxis_title="Soulte",
        template=DARK, height=H, margin=MARGIN,
        hovermode="x unified", showlegend=False
    )
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 4: Z-Spread vs YTM ─────────────────────────────────────────────
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=ytm_grid*100, y=z_spreads, mode="lines", name="Z-Spread",
        line=dict(color=CB, width=2.5),
        hovertemplate="%{x:.2f}% → %{y:.2f} bps<extra></extra>"
    ))
    fig4.add_vline(x=ytm*100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig4.update_layout(
        title="Z-Spread vs YTM",
        xaxis_title="YTM (%)", yaxis_title="Z-Spread (bps)",
        template=DARK, height=H, margin=MARGIN,
        hovermode="x unified", showlegend=False
    )
    fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 5: Modified Duration vs YTM ────────────────────────────────────
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=ytm_grid*100, y=durations, mode="lines", name="Mod. Duration",
        line=dict(color=CA, width=2.5),
        hovertemplate="%{x:.2f}% → %{y:.4f} yr<extra></extra>"
    ))
    fig5.add_vline(x=ytm*100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig5.update_layout(
        title="Modified Duration vs YTM",
        xaxis_title="YTM (%)", yaxis_title="Duration (yr)",
        template=DARK, height=H, margin=MARGIN,
        hovermode="x unified", showlegend=False
    )
    fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig5.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 6: DV01 vs YTM ─────────────────────────────────────────────────
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=ytm_grid*100, y=dv01s, mode="lines", name="DV01",
        line=dict(color=CC, width=2.5),
        hovertemplate="%{x:.2f}% → %{y:.6f}<extra></extra>"
    ))
    fig6.add_vline(x=ytm*100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig6.update_layout(
        title="DV01 vs YTM",
        xaxis_title="YTM (%)", yaxis_title="DV01",
        template=DARK, height=H, margin=MARGIN,
        hovermode="x unified", showlegend=False
    )
    fig6.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig6.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
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
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.plotly_chart(fig1, use_container_width=True, key="fig1")
    with col_b:
        st.plotly_chart(fig2, use_container_width=True, key="fig2")
    with col_c:
        st.plotly_chart(fig3, use_container_width=True, key="fig3")
    
    st.markdown("#### 2. Risk Metrics")
    col_d, col_e, col_f = st.columns(3)
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
