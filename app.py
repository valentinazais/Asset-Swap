# run: streamlit run app.py

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap", layout="wide", initial_sidebar_state="collapsed")



# ── MATH ──────────────────────────────────────────────────────────────────────

def discount_factors(r: float, T: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, int(round(T * freq)))
    t = np.arange(1, n + 1) / freq
    return t, np.exp(-r * t)

def bond_dirty_price(c: float, T: float, y: float, F: float = 100.0, freq: int = 2) -> float:
    t, df = discount_factors(y, T, freq)
    cf = np.full(len(t), c / freq * F)
    cf[-1] += F
    return float(np.dot(cf, df))

def par_asw_spread(c: float, T: float, y: float, r: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Par ASW spread: investor pays dirty price, receives par.
    Soulte = P - F compensated by spread s on floating leg.
    s = (c - r_par) where r_par is the par coupon at risk-free curve,
    but exact formula:
      s * Annuity_rf = (F - P)/F * F  +  (c - c_par) * Annuity_rf
    Standard formula:
      s = [ c*Ann_rf + Z(T) - P/F ] / Ann_rf   (all per unit of face)
    """
    t, df_rf = discount_factors(r, T, freq)
    ann = float(np.sum(df_rf)) / freq          # annuity per unit notional per year
    ZT  = np.exp(-r * T)
    P   = bond_dirty_price(c, T, y, F, freq) / F   # normalised
    if ann < 1e-12:
        return 0.0
    return float((c * ann + ZT - P) / ann) * 1e4

def z_spread_bps(c: float, T: float, y: float, r: float, F: float = 100.0, freq: int = 2) -> float:
    """Z-spread: single parallel shift z to risk-free curve that reprices the bond."""
    dp = bond_dirty_price(c, T, y, F, freq)
    def pv_diff(z):
        t, _ = discount_factors(r, T, freq)
        df = np.exp(-(r + z) * t)
        cf = np.full(len(t), c / freq * F)
        cf[-1] += F
        return float(np.dot(cf, df)) - dp
    try:
        return brentq(pv_diff, -0.5, 5.0, xtol=1e-12) * 1e4
    except Exception:
        return float("nan")

def yield_asw_bps(y: float, r: float) -> float:
    return (y - r) * 1e4

def modified_duration(c: float, T: float, y: float, F: float = 100.0, freq: int = 2) -> float:
    t, df = discount_factors(y, T, freq)
    cf = np.full(len(t), c / freq * F)
    cf[-1] += F
    P = float(np.dot(cf, df))
    D_mac = float(np.dot(t * cf, df)) / P
    return D_mac / (1 + y / freq)

def all_metrics(c, T, y, r, F, freq):
    P   = bond_dirty_price(c, T, y, F, freq)
    sol = P - F
    pa  = par_asw_spread(c, T, y, r, F, freq)
    zs  = z_spread_bps(c, T, y, r, F, freq)
    ya  = yield_asw_bps(y, r)
    md  = modified_duration(c, T, y, F, freq)
    dv01 = P * md / 1e4
    t, df_rf = discount_factors(r, T, freq)
    ann  = float(np.sum(df_rf)) / freq
    fixed_pv = float(np.sum(c / freq * F * df_rf) + F * np.exp(-r * T))
    return dict(P=P, sol=sol, pa=pa, zs=zs, ya=ya, md=md, dv01=dv01, ann=ann, fixed_pv=fixed_pv)

# ── CONTROLS ──────────────────────────────────────────────────────────────────

st.markdown("### Asset Swap Pricer")

if "rc" not in st.session_state:
    st.session_state.rc = 0
rc = st.session_state.rc

cols = st.columns([1.2, 1.2, 1.2, 1.0, 1.2, 1.2, 0.6])
face    = cols[0].number_input("Face",         value=100.0,  step=10.0,   format="%.0f",  key=f"F_{rc}")
coupon  = cols[1].number_input("Coupon",       value=0.050,  step=0.005,  format="%.3f",  min_value=0.001, max_value=0.25,  key=f"c_{rc}")
mat     = cols[2].number_input("Maturity (yr)",value=5.0,    step=0.5,    format="%.1f",  min_value=0.5,   max_value=30.0,  key=f"T_{rc}")
freq    = cols[3].selectbox("Freq", [1,2,4], index=1, format_func=lambda x:{1:"Ann",2:"Semi",4:"Qtr"}[x], key=f"fr_{rc}")
ytm     = cols[4].number_input("YTM",          value=0.055,  step=0.005,  format="%.3f",  min_value=0.001, max_value=0.30,  key=f"y_{rc}")
rf      = cols[5].number_input("Risk-Free",    value=0.030,  step=0.005,  format="%.3f",  min_value=0.001, max_value=0.20,  key=f"r_{rc}")
with cols[6]:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻", help="Reset"):
        st.session_state.rc += 1
        st.rerun()

m = all_metrics(coupon, mat, ytm, rf, face, freq)

# ── METRICS ───────────────────────────────────────────────────────────────────

mc = st.columns(8)
mc[0].metric("Dirty Price",    f"{m['P']:.3f}")
mc[1].metric("Soulte",         f"{m['sol']:+.3f}")
mc[2].metric("Par ASW (bps)",  f"{m['pa']:.2f}")
mc[3].metric("Z-Spread (bps)", f"{m['zs']:.2f}")
mc[4].metric("Yield ASW (bps)",f"{m['ya']:.2f}")
mc[5].metric("Mod. Duration",  f"{m['md']:.4f}")
mc[6].metric("DV01",           f"{m['dv01']:.5f}")
mc[7].metric("Annuity",        f"{m['ann']:.4f}")

st.divider()

# ── SWEEP GRIDS ───────────────────────────────────────────────────────────────

N = 200
ytm_g = np.linspace(max(0.002, ytm - 0.06), ytm + 0.08, N)
rf_g  = np.linspace(max(0.001, rf  - 0.04), rf  + 0.08, N)
cpn_g = np.linspace(max(0.005, coupon - 0.04), coupon + 0.06, N)
mat_g = np.linspace(0.5, 20.0, N)

def sweep(param, grid):
    pa, zs, ya, sol = [], [], [], []
    for v in grid:
        kw = dict(c=coupon, T=mat, y=ytm, r=rf, F=face, freq=freq)
        kw[param] = v
        res = all_metrics(**kw)
        pa.append(res["pa"]); zs.append(res["zs"])
        ya.append(res["ya"]); sol.append(res["sol"])
    return np.array(pa), np.array(zs), np.array(ya), np.array(sol)

pa_y, zs_y, ya_y, so_y = sweep("y", ytm_g)
pa_r, zs_r, ya_r, so_r = sweep("r", rf_g)
pa_c, zs_c, ya_c, so_c = sweep("c", cpn_g)
pa_m, zs_m, ya_m, so_m = sweep("T", mat_g)

# ── PLOT HELPERS ──────────────────────────────────────────────────────────────

DARK = "plotly_dark"
H    = 270
CA, CB, CC, CD = "#00b4d8", "#f77f00", "#06d6a0", "#e63946"
MARGIN = dict(t=24, b=32, l=44, r=12)

def spread_chart(x, pa, zs, ya, xlab, vline=None):
    fig = go.Figure()
    fig.add_scatter(x=x, y=pa, name="Par ASW",   line=dict(color=CA, width=2))
    fig.add_scatter(x=x, y=zs, name="Z-Spread",  line=dict(color=CB, width=2, dash="dash"))
    fig.add_scatter(x=x, y=ya, name="Yield ASW", line=dict(color=CC, width=2, dash="dot"))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dot", line_color="white", line_width=1)
    fig.update_layout(template=DARK, height=H, margin=MARGIN,
                      xaxis_title=xlab, yaxis_title="bps",
                      legend=dict(orientation="h", y=1.15, x=0, font_size=11))
    return fig

def soulte_chart(x, so, xlab, vline=None):
    fig = go.Figure()
    fig.add_scatter(x=x, y=so, line=dict(color=CD, width=2), name="Soulte")
    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dot", line_color="white", line_width=1)
    fig.update_layout(template=DARK, height=H, margin=MARGIN,
                      xaxis_title=xlab, yaxis_title="Soulte", showlegend=False)
    return fig

# ── ROW 1 : vs YTM ────────────────────────────────────────────────────────────

r1, r2 = st.columns(2)
with r1:
    st.caption("Spreads — vs YTM")
    st.plotly_chart(spread_chart(ytm_g*100, pa_y, zs_y, ya_y, "YTM (%)", ytm*100), use_container_width=True)
with r2:
    st.caption("Soulte — vs YTM")
    st.plotly_chart(soulte_chart(ytm_g*100, so_y, "YTM (%)", ytm*100), use_container_width=True)

# ── ROW 2 : vs Risk-Free ──────────────────────────────────────────────────────

r3, r4 = st.columns(2)
with r3:
    st.caption("Spreads — vs Risk-Free Rate")
    st.plotly_chart(spread_chart(rf_g*100, pa_r, zs_r, ya_r, "Risk-Free (%)", rf*100), use_container_width=True)
with r4:
    st.caption("Soulte — vs Risk-Free Rate")
    st.plotly_chart(soulte_chart(rf_g*100, so_r, "Risk-Free (%)", rf*100), use_container_width=True)

# ── ROW 3 : vs Coupon & Maturity ──────────────────────────────────────────────

r5, r6, r7, r8 = st.columns(4)
with r5:
    st.caption("Spreads — vs Coupon")
    st.plotly_chart(spread_chart(cpn_g*100, pa_c, zs_c, ya_c, "Coupon (%)", coupon*100), use_container_width=True)
with r6:
    st.caption("Soulte — vs Coupon")
    st.plotly_chart(soulte_chart(cpn_g*100, so_c, "Coupon (%)", coupon*100), use_container_width=True)
with r7:
    st.caption("Spreads — vs Maturity")
    st.plotly_chart(spread_chart(mat_g, pa_m, zs_m, ya_m, "Maturity (yr)", mat), use_container_width=True)
with r8:
    st.caption("Soulte — vs Maturity")
    st.plotly_chart(soulte_chart(mat_g, so_m, "Maturity (yr)", mat), use_container_width=True)

# ── ROW 4 : Duration / Leg PVs / Cashflows ────────────────────────────────────

r9, r10, r11 = st.columns(3)

with r9:
    st.caption("Duration & DV01 — vs YTM")
    durs, dv01s = [], []
    for y_ in ytm_g:
        res = all_metrics(coupon, mat, y_, rf, face, freq)
        durs.append(res["md"]); dv01s.append(res["dv01"])
    fig_d = make_subplots(specs=[[{"secondary_y": True}]])
    fig_d.add_trace(go.Scatter(x=ytm_g*100, y=durs,  name="Mod. Dur",  line=dict(color=CA, width=2)), secondary_y=False)
    fig_d.add_trace(go.Scatter(x=ytm_g*100, y=dv01s, name="DV01",      line=dict(color=CB, width=2, dash="dash")), secondary_y=True)
    fig_d.add_vline(x=ytm*100, line_dash="dot", line_color="white", line_width=1)
    fig_d.update_layout(template=DARK, height=H, margin=MARGIN,
                        legend=dict(orientation="h", y=1.15, x=0, font_size=11))
    fig_d.update_xaxes(title_text="YTM (%)")
    fig_d.update_yaxes(title_text="Duration", secondary_y=False)
    fig_d.update_yaxes(title_text="DV01",     secondary_y=True)
    st.plotly_chart(fig_d, use_container_width=True)

with r10:
    st.caption("Fixed Leg PV — vs YTM")
    fpvs, fpvs_rf = [], []
    for y_ in ytm_g:
        res_y  = all_metrics(coupon, mat, y_,  rf,  face, freq)
        res_rf = all_metrics(coupon, mat, y_,  rf,  face, freq)
        fpvs.append(res_y["fixed_pv"])
    # Also sweep rf for fixed_pv
    fpvs_r = []
    for r_ in rf_g:
        res = all_metrics(coupon, mat, ytm, r_, face, freq)
        fpvs_r.append(res["fixed_pv"])
    fig_l = go.Figure()
    fig_l.add_scatter(x=ytm_g*100, y=fpvs,  name="Fixed PV (ytm axis)", line=dict(color=CA, width=2))
    fig_l.add_hline(y=face, line_dash="dot", line_color="grey", line_width=1)
    fig_l.add_vline(x=ytm*100, line_dash="dot", line_color="white", line_width=1)
    fig_l.update_layout(template=DARK, height=H, margin=MARGIN,
                        xaxis_title="YTM (%)", yaxis_title="PV",
                        legend=dict(orientation="h", y=1.15, x=0, font_size=11))
    st.plotly_chart(fig_l, use_container_width=True)

with r11:
    st.caption("Cashflow Schedule")
    n_cf = max(1, int(round(mat * freq)))
    t_cf = np.arange(1, n_cf + 1) / freq
    cpn_cf = np.full(n_cf, coupon / freq * face)
    prin   = np.zeros(n_cf); prin[-1] = face
    fig_cf = go.Figure()
    fig_cf.add_bar(x=t_cf, y=cpn_cf, name="Coupon",    marker_color=CA)
    fig_cf.add_bar(x=t_cf, y=prin,   name="Principal", marker_color=CB)
    fig_cf.update_layout(barmode="stack", template=DARK, height=H, margin=MARGIN,
                         xaxis_title="Time (yr)", yaxis_title="CF",
                         legend=dict(orientation="h", y=1.15, x=0, font_size=11))
    st.plotly_chart(fig_cf, use_container_width=True)

# ── ROW 5 : 3D Surface ────────────────────────────────────────────────────────

st.divider()
st.caption("Par ASW Surface — Maturity × YTM")

@st.cache_data(show_spinner=False)
def _surface(c, r_, F, freq_):
    mats = np.linspace(1, 15, 40)
    ytms = np.linspace(max(0.002, r_ - 0.03), r_ + 0.10, 40)
    Z = np.zeros((len(mats), len(ytms)))
    for i, m_ in enumerate(mats):
        for j, y_ in enumerate(ytms):
            Z[i, j] = par_asw_spread(c, m_, y_, r_, F, freq_)
    return mats, ytms, Z

mats_s, ytms_s, Z_s = _surface(coupon, rf, face, freq)
fig_s = go.Figure(go.Surface(
    x=ytms_s*100, y=mats_s, z=Z_s,
    colorscale="Viridis",
    contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True))
))
fig_s.update_layout(
    scene=dict(xaxis_title="YTM (%)", yaxis_title="Maturity (yr)", zaxis_title="Par ASW (bps)"),
    template=DARK, height=500, margin=dict(t=10, b=0, l=0, r=0),
)
st.plotly_chart(fig_s, use_container_width=True)
