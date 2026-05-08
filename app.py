from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap", layout="wide", initial_sidebar_state="collapsed")

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
    .input-label { font-size: 12px; color: #aaa; font-weight: 600; margin-bottom: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("Asset Swap Pricer")

# ── INITIALIZE SESSION STATE ──────────────────────────────────────────────────

if "face" not in st.session_state:
    st.session_state.face = 100.0
if "coupon_pct" not in st.session_state:
    st.session_state.coupon_pct = 6.5
if "mat" not in st.session_state:
    st.session_state.mat = 7.0
if "ytm_pct" not in st.session_state:
    st.session_state.ytm_pct = 8.0
if "rf_pct" not in st.session_state:
    st.session_state.rf_pct = 3.0
if "freq_label" not in st.session_state:
    st.session_state.freq_label = "Semi"

# ── INPUT CONTROLS ────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5, col6 = st.columns(6)

# FACE
with col1:
    st.markdown('<div class="input-label">Face</div>', unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("−", key="face_minus"):
            st.session_state.face = max(50.0, st.session_state.face - 5.0)
    with btn_col2:
        if st.button("+", key="face_plus"):
            st.session_state.face = min(500.0, st.session_state.face + 5.0)
    st.session_state.face = st.slider("##face", 50.0, 500.0, st.session_state.face, 1.0, label_visibility="collapsed")

# COUPON
with col2:
    st.markdown('<div class="input-label">Coupon (%)</div>', unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("−", key="coupon_minus"):
            st.session_state.coupon_pct = max(0.5, st.session_state.coupon_pct - 0.25)
    with btn_col2:
        if st.button("+", key="coupon_plus"):
            st.session_state.coupon_pct = min(15.0, st.session_state.coupon_pct + 0.25)
    st.session_state.coupon_pct = st.slider("##coupon", 0.5, 15.0, st.session_state.coupon_pct, 0.05, label_visibility="collapsed")

# MATURITY
with col3:
    st.markdown('<div class="input-label">Maturity (yr)</div>', unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("−", key="mat_minus"):
            st.session_state.mat = max(0.5, st.session_state.mat - 0.5)
    with btn_col2:
        if st.button("+", key="mat_plus"):
            st.session_state.mat = min(30.0, st.session_state.mat + 0.5)
    st.session_state.mat = st.slider("##mat", 0.5, 30.0, st.session_state.mat, 0.25, label_visibility="collapsed")

# FREQUENCY
with col4:
    st.markdown('<div class="input-label">Freq</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.freq_label = st.selectbox("##freq", ["Annual", "Semi", "Quarterly"], index=["Annual", "Semi", "Quarterly"].index(st.session_state.freq_label), label_visibility="collapsed")

# YTM
with col5:
    st.markdown('<div class="input-label">YTM (%)</div>', unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("−", key="ytm_minus"):
            st.session_state.ytm_pct = max(0.5, st.session_state.ytm_pct - 0.25)
    with btn_col2:
        if st.button("+", key="ytm_plus"):
            st.session_state.ytm_pct = min(20.0, st.session_state.ytm_pct + 0.25)
    st.session_state.ytm_pct = st.slider("##ytm", 0.5, 20.0, st.session_state.ytm_pct, 0.05, label_visibility="collapsed")

# RISK-FREE
with col6:
    st.markdown('<div class="input-label">Risk-Free (%)</div>', unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("−", key="rf_minus"):
            st.session_state.rf_pct = max(0.1, st.session_state.rf_pct - 0.25)
    with btn_col2:
        if st.button("+", key="rf_plus"):
            st.session_state.rf_pct = min(15.0, st.session_state.rf_pct + 0.25)
    st.session_state.rf_pct = st.slider("##rf", 0.1, 15.0, st.session_state.rf_pct, 0.05, label_visibility="collapsed")

# ── EXTRACT VALUES ────────────────────────────────────────────────────────────

face = st.session_state.face
coupon_pct = st.session_state.coupon_pct
coupon = coupon_pct / 100
mat = st.session_state.mat
freq_map = {"Annual": 1, "Semi": 2, "Quarterly": 4}
freq = freq_map[st.session_state.freq_label]
ytm_pct = st.session_state.ytm_pct
ytm = ytm_pct / 100
rf_pct = st.session_state.rf_pct
rf = rf_pct / 100

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
    
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    
    with m1:
        st.metric("Dirty Price", f"{P:.4f}")
    with m2:
        st.metric("Soulte", f"{slt:.4f}")
    with m3:
        st.metric("Par ASW (bps)", f"{par_asw_val:.2f}")
    with m4:
        st.metric("Z-Spread (bps)", f"{z_sp:.2f}")
    with m5:
        st.metric("Yield ASW (bps)", f"{y_asw:.2f}")
    with m6:
        st.metric("Mod Duration (yr)", f"{md:.3f}")
    with m7:
        st.metric("DV01", f"{dv01_val*100:.3f}%")
    with m8:
        st.metric("Annuity", f"{ann:.3f}")
    
    
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
    fig1.add_trace(go.Scatter(x=ytm_grid*100, y=prices, mode="lines", name="Bond Price",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.2f}<extra></extra>"))
    fig1.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig1.update_layout(template=DARK, height=H, margin=MARGIN, title="Bond Price vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Price (%)", hovermode="x unified", showlegend=False)
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 2: Par ASW vs YTM ──────────────────────────────────────────────
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ytm_grid*100, y=par_asws, mode="lines", name="Par ASW",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.1f} bps<extra></extra>"))
    fig2.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig2.update_layout(template=DARK, height=H, margin=MARGIN, title="Par ASW vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Par ASW (bps)", hovermode="x unified", showlegend=False)
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 3: Soulte vs YTM ───────────────────────────────────────────────
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ytm_grid*100, y=soultes, mode="lines", name="Soulte",
                     line=dict(color=CC, width=2.5), hovertemplate="%{x:.2f}% → %{y:.4f}<extra></extra>"))
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, annotation_text="At Par", annotation_position="right")
    fig3.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig3.update_layout(template=DARK, height=H, margin=MARGIN, title="Soulte vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Soulte (price)", hovermode="x unified", showlegend=False)
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 4: Z-Spread vs YTM ─────────────────────────────────────────────
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ytm_grid*100, y=z_spreads, mode="lines", name="Z-Spread",
                     line=dict(color=CB, width=2.5), hovertemplate="%{x:.2f}% → %{y:.1f} bps<extra></extra>"))
    fig4.add_vline(x=ytm_pct, line_dash="dash", line_color=CA, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig4.update_layout(template=DARK, height=H, margin=MARGIN, title="Z-Spread vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Z-Spread (bps)", hovermode="x unified", showlegend=False)
    fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 5: Modified Duration vs YTM ────────────────────────────────────
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=ytm_grid*100, y=durations, mode="lines", name="Mod. Duration",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.4f} yr<extra></extra>"))
    fig5.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig5.update_layout(template=DARK, height=H, margin=MARGIN, title="Modified Duration vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Duration (yr)", hovermode="x unified", showlegend=False)
    fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig5.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 6: DV01 vs YTM ─────────────────────────────────────────────────
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=ytm_grid*100, y=dv01s, mode="lines", name="DV01",
                     line=dict(color=CC, width=2.5), hovertemplate="%{x:.2f}% → %{y:.6f}<extra></extra>"))
    fig6.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig6.update_layout(template=DARK, height=H, margin=MARGIN, title="DV01 vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="DV01", hovermode="x unified", showlegend=False)
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
