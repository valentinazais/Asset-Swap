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
        return 0.0
    
    numerator = P - df_terminal
    denominator = ann_rf
    
    asw_spread = numerator / denominator
    return asw_spread * 10000


def yield_asw(coupon: float, T: float, ytm: float, rf: float, asw_bps: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate yield asset swap spread."""
    return asw_bps - (ytm - rf) * 10000


def mod_duration(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate modified duration."""
    shift = 0.0001
    P0 = price_dirty(coupon, T, ytm, F, freq)
    P_up = price_dirty(coupon, T, ytm + shift, F, freq)
    P_dn = price_dirty(coupon, T, ytm - shift, F, freq)
    
    duration = (P_dn - P_up) / (2 * P0 * shift)
    return duration / (1 + ytm / freq)


def dv01(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """Calculate DV01 per $100 face."""
    shift = 0.0001
    P0 = price_dirty(coupon, T, ytm, F, freq)
    P_up = price_dirty(coupon, T, ytm + shift, F, freq)
    
    return abs(P_up - P0)


# ── UI ────────────────────────────────────────────────────────────────────────

try:
    st.title("Asset Swap Pricer")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        face = st.number_input("Face", value=100.0, step=1.0, format="%.2f", label_visibility="collapsed")
    
    with col2:
        coupon = st.slider("Coupon (%)", 0.0, 15.0, 6.45, 0.05, label_visibility="collapsed") / 100
    
    with col3:
        T = st.slider("Maturity (yr)", 0.25, 30.0, 7.0, 0.25, label_visibility="collapsed")
    
    with col4:
        freq_name = st.selectbox("Freq", ["Annual", "Semi", "Quarterly"], index=1, label_visibility="collapsed")
        freq = {"Annual": 1, "Semi": 2, "Quarterly": 4}[freq_name]
    
    with col5:
        ytm = st.slider("YTM (%)", 0.1, 15.0, 8.0, 0.05, label_visibility="collapsed") / 100
    
    with col6:
        rf = st.slider("Risk-Free (%)", 0.1, 15.0, 5.95, 0.05, label_visibility="collapsed") / 100
    
    # ── CALCULATIONS ──────────────────────────────────────────────────────────
    
    P = price_dirty(coupon, T, ytm, face, freq)
    soulte = P - face
    asw_par = par_asw(coupon, T, ytm, rf, face, freq)
    z_spread = 500.0
    y_asw = yield_asw(coupon, T, ytm, rf, z_spread, face, freq)
    mod_dur = mod_duration(coupon, T, ytm, face, freq)
    dv01_val = dv01(coupon, T, ytm, face, freq)
    ann = annuity_factor(ytm, T, freq)
    
    # ── METRICS ROW ───────────────────────────────────────────────────────────
    
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6, col_m7, col_m8 = st.columns(8)
    
    with col_m1:
        st.metric("Dirty Price", f"{P:.4f}")
    with col_m2:
        st.metric("Soulte", f"{soulte:.4f}")
    with col_m3:
        st.metric("Par ASW (bps)", f"{asw_par:.2f}")
    with col_m4:
        st.metric("Z-Spread (bps)", f"{z_spread:.2f}")
    with col_m5:
        st.metric("Yield ASW (bps)", f"{y_asw:.2f}")
    with col_m6:
        st.metric("Mod. Duration", f"{mod_dur:.3f} yr")
    with col_m7:
        st.metric("DV01", f"{dv01_val*100:.3f}%")
    with col_m8:
        st.metric("Annuity", f"{ann:.3f}")
    
    # ── FIGURE 1: Price vs YTM ────────────────────────────────────────────────
    
    ytm_grid = np.linspace(0.001, 0.15, 100)
    prices = np.array([price_dirty(coupon, T, y_, face, freq) for y_ in ytm_grid])
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ytm_grid*100, y=prices, mode="lines", name="Price", 
                              line=dict(color=CA, width=2.5), hovertemplate="%{y:.4f}<extra></extra>"))
    fig1.add_vline(x=ytm*100, line=dict(color=CB, width=1.5, dash="dash"), annotation_text="YTM")
    fig1.update_layout(template=DARK, height=H, margin=MARGIN, title="Bond Price vs YTM",
                       xaxis_title="YTM (%)", yaxis_title="Price", hovermode="x unified", showlegend=False)
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 2: Par ASW vs YTM ──────────────────────────────────────────────
    
    asw_vals = np.array([par_asw(coupon, T, y_, rf, face, freq) for y_ in ytm_grid])
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ytm_grid*100, y=asw_vals, mode="lines", name="Par ASW",
                              line=dict(color=CA, width=2.5), hovertemplate="%{y:.2f} bps<extra></extra>"))
    fig2.add_hline(y=0, line=dict(color="white", width=0.5, dash="dash"))
    fig2.add_vline(x=ytm*100, line=dict(color=CB, width=1.5, dash="dash"), annotation_text="YTM")
    fig2.update_layout(template=DARK, height=H, margin=MARGIN, title="Par ASW vs YTM",
                       xaxis_title="YTM (%)", yaxis_title="Par ASW (bps)", hovermode="x unified", showlegend=False)
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 3: Soulte vs YTM ───────────────────────────────────────────────
    
    soultes = np.array([price_dirty(coupon, T, y_, face, freq) - face for y_ in ytm_grid])
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ytm_grid*100, y=soultes, mode="lines", name="Soulte",
                              line=dict(color=CC, width=2.5), hovertemplate="%{y:.4f}<extra></extra>"))
    fig3.add_hline(y=0, line=dict(color="white", width=1, dash="dash"), annotation_text="At Par")
    fig3.add_vline(x=ytm*100, line=dict(color=CB, width=1.5, dash="dash"), annotation_text="YTM")
    fig3.update_layout(template=DARK, height=H, margin=MARGIN, title="Soulte vs YTM",
                       xaxis_title="YTM (%)", yaxis_title="Soulte (price)", hovermode="x unified", showlegend=False)
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 4: Modified Duration vs YTM ────────────────────────────────────
    
    durations = np.array([mod_duration(coupon, T, y_, face, freq) for y_ in ytm_grid])
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ytm_grid*100, y=durations, mode="lines", name="Mod. Duration",
                              line=dict(color=CA, width=2.5), hovertemplate="%{y:.4f} yr<extra></extra>"))
    fig4.add_vline(x=ytm*100, line=dict(color=CB, width=1.5, dash="dash"), annotation_text="YTM")
    fig4.update_layout(template=DARK, height=H, margin=MARGIN, title="Mod. Duration vs YTM",
                       xaxis_title="YTM (%)", yaxis_title="Duration (yr)", hovermode="x unified", showlegend=False)
    fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 5: Par ASW vs Maturity ─────────────────────────────────────────
    
    mat_grid = np.linspace(0.25, 30, 100)
    asw_mats = np.array([par_asw(coupon, m_, ytm, rf, face, freq) for m_ in mat_grid])
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=mat_grid, y=asw_mats, mode="lines", name="Par ASW",
                              line=dict(color=CA, width=2.5), hovertemplate="%{y:.2f} bps<extra></extra>"))
    fig5.add_hline(y=0, line=dict(color="white", width=0.5, dash="dash"))
    fig5.add_vline(x=T, line=dict(color=CB, width=1.5, dash="dash"), annotation_text="Maturity")
    fig5.update_layout(template=DARK, height=H, margin=MARGIN, title="Par ASW vs Maturity",
                       xaxis_title="Maturity (yr)", yaxis_title="Par ASW (bps)", hovermode="x unified", showlegend=False)
    fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig5.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 6: DV01 vs YTM ─────────────────────────────────────────────────
    
    dv01s = np.array([dv01(coupon, T, y_, face, freq) for y_ in ytm_grid])
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=ytm_grid*100, y=dv01s*100, mode="lines", name="DV01",
                              line=dict(color=CA, width=2.5), hovertemplate="%{y:.3f}%<extra></extra>"))
    fig6.add_vline(x=ytm*100, line=dict(color=CB, width=1.5, dash="dash"), annotation_text="YTM")
    fig6.update_layout(template=DARK, height=H, margin=MARGIN, title="DV01 vs YTM",
                      xaxis_title="YTM (%)", yaxis_title="DV01 (%)", hovermode="x unified", showlegend=False)
    fig6.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig6.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    
    # ── FIGURE 7: Par ASW Surface (Maturity × YTM) ────────────────────────────
    
    mat_grid_s = np.linspace(0.5, 30, 50)
    ytm_grid_s = np.linspace(0.001, 0.15, 50)
    Z_s = np.array([
        [par_asw(coupon, m_, y_, rf, face, freq) for y_ in ytm_grid_s]
        for m_ in mat_grid_s
    ])
    
    fig7 = go.Figure(go.Surface(
        x=ytm_grid_s*100, y=mat_grid_s, z=Z_s,
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
