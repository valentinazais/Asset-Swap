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
    
    if D_mac < 1e-12 or ytm < 1e-12:
        return float("nan")
    
    return D_mac / (1 + ytm)


def dv01(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate DV01 (price change per 1 bp)."""
    D_mod = modified_duration(coupon, T, ytm, F, freq)
    P = price_dirty(coupon, T, ytm, F, freq)
    
    if P < 1e-12 or np.isnan(D_mod):
        return float("nan")
    
    return D_mod * P / 10000


# ── STATE ─────────────────────────────────────────────────────────────────────

if "face" not in st.session_state:
    st.session_state.face = 100.0
if "coupon" not in st.session_state:
    st.session_state.coupon = 6.45
if "maturity" not in st.session_state:
    st.session_state.maturity = 7.0
if "freq_idx" not in st.session_state:
    st.session_state.freq_idx = 0
if "ytm_pct" not in st.session_state:
    st.session_state.ytm_pct = 8.0
if "rf_pct" not in st.session_state:
    st.session_state.rf_pct = 5.95

# ── INPUTS ────────────────────────────────────────────────────────────────────

st.markdown("# Asset Swap Pricer")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown("**Face**")
    st.session_state.face = st.slider("##face", 1.0, 1000.0, st.session_state.face, 1.0, label_visibility="collapsed")

with col2:
    st.markdown("**Coupon (%)**")
    st.session_state.coupon = st.slider("##coupon", 0.01, 15.0, st.session_state.coupon, 0.01, label_visibility="collapsed")

with col3:
    st.markdown("**Maturity (yr)**")
    st.session_state.maturity = st.slider("##maturity", 0.1, 30.0, st.session_state.maturity, 0.1, label_visibility="collapsed")

with col4:
    st.markdown("**Freq**")
    freq_opts = ("Annual", "Semi", "Quarterly", "Monthly")
    st.session_state.freq_idx = st.selectbox("##freq", range(len(freq_opts)), st.session_state.freq_idx,
                                             format_func=lambda i: freq_opts[i], label_visibility="collapsed")

with col5:
    st.markdown("**YTM (%)**")
    st.session_state.ytm_pct = st.slider("##ytm", 0.1, 15.0, st.session_state.ytm_pct, 0.05, label_visibility="collapsed")

with col6:
    st.markdown("**Risk-Free (%)**")
    st.session_state.rf_pct = st.slider("##rf", 0.1, 15.0, st.session_state.rf_pct, 0.05, label_visibility="collapsed")

# ── MAPPING ───────────────────────────────────────────────────────────────────

face = st.session_state.face
coupon = st.session_state.coupon / 100
T = st.session_state.maturity
freq_map = {0: 1, 1: 2, 2: 4, 3: 12}
freq = freq_map[st.session_state.freq_idx]
ytm = st.session_state.ytm_pct / 100
rf = st.session_state.rf_pct / 100
ytm_pct = st.session_state.ytm_pct

# ── COMPUTE ───────────────────────────────────────────────────────────────────

try:
    P_dirty = price_dirty(coupon, T, ytm, face, freq)
    slt = soulte(coupon, T, ytm, face, freq)
    p_asw = par_asw(coupon, T, ytm, rf, face, freq)
    z_spr = z_spread(coupon, T, ytm, rf, face, freq)
    y_asw = yield_asw(coupon, T, ytm, rf, face, freq)
    mod_dur = modified_duration(coupon, T, ytm, face, freq)
    dv01_val = dv01(coupon, T, ytm, face, freq)
    ann = annuity_factor(ytm, T, freq)
    
    # ── METRICS ROW ───────────────────────────────────────────────────────────
    
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    
    with m1:
        st.metric("Dirty Price", f"{P_dirty:.4f}")
    with m2:
        st.metric("Soulte", f"{slt:.4f}")
    with m3:
        st.metric("Par ASW (bps)", f"{p_asw:.2f}")
    with m4:
        st.metric("Z-Spread (bps)", f"{z_spr:.2f}")
    with m5:
        st.metric("Yield ASW (bps)", f"{y_asw:.2f}")
    with m6:
        st.metric("Mod. Duration", f"{mod_dur:.3f} yr")
    with m7:
        st.metric("DV01", f"{dv01_val*100:.3f}%")
    with m8:
        st.metric("Annuity", f"{ann:.3f}")
    
    # ── FIGURE 1: Bond Price vs YTM ────────────────────────────────────────────
    
    ytm_grid = np.linspace(0.001, 0.15, 100)
    prices = np.array([price_dirty(coupon, T, y_, face, freq) for y_ in ytm_grid])
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ytm_grid*100, y=prices, mode="lines", name="Price",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.4f}<extra></extra>"))
    fig1.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig1.update_layout(template=DARK, height=H, margin=MARGIN, title="Bond Price vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Price (%)", hovermode="x unified", showlegend=False)
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 2: Par ASW vs YTM ──────────────────────────────────────────────
    
    par_asws = np.array([par_asw(coupon, T, y_, rf, face, freq) for y_ in ytm_grid])
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ytm_grid*100, y=par_asws, mode="lines", name="Par ASW",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.2f} bps<extra></extra>"))
    fig2.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig2.update_layout(template=DARK, height=H, margin=MARGIN, title="Par ASW vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Par ASW (bps)", hovermode="x unified", showlegend=False)
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 3: Soulte vs YTM ───────────────────────────────────────────────
    
    soultes = np.array([soulte(coupon, T, y_, face, freq) for y_ in ytm_grid])
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ytm_grid*100, y=soultes, mode="lines", name="Soulte",
                     line=dict(color=CC, width=2.5), hovertemplate="%{x:.2f}% → %{y:.4f}<extra></extra>"))
    fig3.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", annotation_text="At Par", annotation_position="right")
    fig3.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig3.update_layout(template=DARK, height=H, margin=MARGIN, title="Soulte vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Soulte (price)", hovermode="x unified", showlegend=False)
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 4: Z-Spread vs YTM ─────────────────────────────────────────────
    
    z_spreads = np.array([z_spread(coupon, T, y_, rf, face, freq) for y_ in ytm_grid])
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ytm_grid*100, y=z_spreads, mode="lines", name="Z-Spread",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.2f} bps<extra></extra>"))
    fig4.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig4.update_layout(template=DARK, height=H, margin=MARGIN, title="Z-Spread vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Z-Spread (bps)", hovermode="x unified", showlegend=False)
    fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 5: Modified Duration vs YTM ────────────────────────────────────
    
    durations = np.array([modified_duration(coupon, T, y_, face, freq) for y_ in ytm_grid])
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=ytm_grid*100, y=durations, mode="lines", name="Mod. Duration",
                     line=dict(color=CA, width=2.5), hovertemplate="%{x:.2f}% → %{y:.4f} yr<extra></extra>"))
    fig5.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig5.update_layout(template=DARK, height=H, margin=MARGIN, title="Modified Duration vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="Duration (yr)", hovermode="x unified", showlegend=False)
    fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig5.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 6: DV01 vs YTM ─────────────────────────────────────────────────
    
    dv01s = np.array([dv01(coupon, T, y_, face, freq) for y_ in ytm_grid])
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=ytm_grid*100, y=dv01s*100, mode="lines", name="DV01",
                     line=dict(color=CC, width=2.5), hovertemplate="%{x:.2f}% → %{y:.3f}%<extra></extra>"))
    fig6.add_vline(x=ytm_pct, line_dash="dash", line_color=CB, line_width=1.5, annotation_text="YTM", annotation_position="top right")
    fig6.update_layout(template=DARK, height=H, margin=MARGIN, title="DV01 vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="DV01 (%)", hovermode="x unified", showlegend=False)
    fig6.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig6.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 7: Par ASW Surface (Maturity × YTM) ────────────────────────────
    
    @st.cache_data(show_spinner=False, ttl=3600)
    def _surface(c, r_, F, freq_):
        mat_grid = np.linspace(0.5, 30, 50)
        ytm_grid_s = np.linspace(0.001, 0.15, 50)
        Z = np.array([
            [par_asw(c, m_, y_, r_, F, freq_) for y_ in ytm_grid_s]
            for m_ in mat_grid
        ])
        return mat_grid, ytm_grid_s, Z
    
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
