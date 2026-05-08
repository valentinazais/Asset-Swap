# run: streamlit run app.py

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
div[data-testid="metric-container"] {background:#1e1e2e;border-radius:6px;padding:6px 10px;}
div[data-testid="stVerticalBlock"] {gap:0.3rem;}
.block-container {padding-top:0.8rem;padding-bottom:0rem;}
</style>
""", unsafe_allow_html=True)

# ── MATH ──────────────────────────────────────────────────────────────────────

def bond_price(c, T, y, F=100.0, freq=2):
    n = max(1, int(round(T * freq)))
    t = np.arange(1, n + 1) / freq
    df = np.exp(-y * t)
    return float(np.sum(c / freq * F * df) + F * df[-1])

def par_asw(c, T, y, r, F=100.0, freq=2):
    n = max(1, int(round(T * freq)))
    t = np.arange(1, n + 1) / freq
    df = np.exp(-r * t)
    ann = float(np.sum(df / freq))
    ZT  = np.exp(-r * T)
    P   = bond_price(c, T, y, F, freq) / F
    return float((c * ann + ZT - P) / ann * 1e4) if ann > 1e-12 else 0.0

def z_spread(c, T, dp, r, F=100.0, freq=2):
    def pv(s):
        n = max(1, int(round(T * freq)))
        t = np.arange(1, n + 1) / freq
        df = np.exp(-(r + s) * t)
        return float(np.sum(c / freq * F * df) + F * df[-1]) - dp
    try:
        return brentq(pv, -0.2, 2.0, xtol=1e-10) * 1e4
    except Exception:
        return float("nan")

def metrics(c, T, y, r, F=100.0, freq=2):
    n = max(1, int(round(T * freq)))
    t = np.arange(1, n + 1) / freq
    df_r = np.exp(-r * t)
    dp   = bond_price(c, T, y, F, freq)
    ann  = float(np.sum(df_r / freq))
    dur  = float(np.sum(t * c / freq * F * df_r) + T * F * np.exp(-r * T)) / dp
    return {
        "dirty": dp,
        "soulte": dp - F,
        "par_asw": par_asw(c, T, y, r, F, freq),
        "z_spd": z_spread(c, T, dp, r, F, freq),
        "y_asw": (y - r) * 1e4,
        "dur": dur,
        "dv01": dp * dur / 1e4,
        "fixed_pv": float(np.sum(c / freq * F * df_r) + F * np.exp(-r * T)),
        "float_pv": F,
        "ann": ann,
    }

# ── CONTROLS ──────────────────────────────────────────────────────────────────

st.markdown("## Asset Swap Pricer")

col_inputs = st.columns(7)

with col_inputs[0]:
    face = st.number_input("Face", value=100.0, step=10.0, format="%.0f")
with col_inputs[1]:
    coupon = st.number_input("Coupon", value=0.050, step=0.005, format="%.3f", min_value=0.001, max_value=0.20)
with col_inputs[2]:
    maturity = st.number_input("Maturity (yr)", value=5.0, step=0.5, min_value=0.5, max_value=30.0, format="%.1f")
with col_inputs[3]:
    freq = st.selectbox("Freq", [1, 2, 4], index=1, format_func=lambda x: {1:"Ann",2:"Semi",4:"Qtrly"}[x])
with col_inputs[4]:
    ytm = st.number_input("YTM", value=0.055, step=0.005, format="%.3f", min_value=0.001, max_value=0.25)
with col_inputs[5]:
    rf = st.number_input("Risk-Free", value=0.030, step=0.005, format="%.3f", min_value=0.0, max_value=0.20)
with col_inputs[6]:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻ Reset"):
        st.rerun()

r = metrics(coupon, maturity, ytm, rf, face, freq)

# ── METRICS ROW ───────────────────────────────────────────────────────────────

m = st.columns(8)
m[0].metric("Dirty Price",   f"{r['dirty']:.3f}")
m[1].metric("Soulte",        f"{r['soulte']:+.3f}")
m[2].metric("Par ASW (bps)", f"{r['par_asw']:.2f}")
m[3].metric("Z-Spread (bps)",f"{r['z_spd']:.2f}")
m[4].metric("Yield ASW (bps)",f"{r['y_asw']:.2f}")
m[5].metric("Mod. Duration", f"{r['dur']:.4f}")
m[6].metric("DV01",          f"{r['dv01']:.6f}")
m[7].metric("Annuity",       f"{r['ann']:.6f}")

st.markdown("---")

# ── PLOTS ─────────────────────────────────────────────────────────────────────

DARK = "plotly_dark"
C    = ["#00b4d8","#f77f00","#06d6a0","#e63946","#a8dadc","#c77dff"]
H    = 290

def _fig():
    return go.Figure()

# Pre-compute grids
ytm_g  = np.linspace(max(0.001, coupon - 0.08), coupon + 0.08, 200)
rf_g   = np.linspace(0.001, 0.15, 200)
cpn_g  = np.linspace(0.005, 0.15, 200)
mat_g  = np.linspace(0.5, 20, 150)

def sweep(param, grid, base):
    pa, zs, ya, so = [], [], [], []
    for v in grid:
        kw = {**base, param: v}
        res = metrics(kw["c"], kw["T"], kw["y"], kw["r"], kw["F"], kw["freq"])
        pa.append(res["par_asw"]); zs.append(res["z_spd"])
        ya.append(res["y_asw"]);   so.append(res["soulte"])
    return pa, zs, ya, so

base = dict(c=coupon, T=maturity, y=ytm, r=rf, F=face, freq=freq)

pa_ytm, zs_ytm, ya_ytm, so_ytm = sweep("y",    ytm_g, base)
pa_rf,  zs_rf,  ya_rf,  so_rf  = sweep("r",    rf_g,  base)
pa_cpn, zs_cpn, ya_cpn, so_cpn = sweep("c",    cpn_g, base)
pa_mat, zs_mat, ya_mat, so_mat = sweep("T",    mat_g, base)

def spread_fig(x, pa, zs, ya, xlab):
    fig = go.Figure()
    fig.add_scatter(x=x, y=pa, name="Par ASW",   line=dict(color=C[0]))
    fig.add_scatter(x=x, y=zs, name="Z-Spread",  line=dict(color=C[1], dash="dash"))
    fig.add_scatter(x=x, y=ya, name="Yield ASW", line=dict(color=C[2], dash="dot"))
    fig.update_layout(xaxis_title=xlab, yaxis_title="bps", template=DARK,
                      height=H, margin=dict(t=28,b=30,l=40,r=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig

def soulte_fig(x, so, xlab):
    fig = go.Figure()
    fig.add_scatter(x=x, y=so, mode="lines", name="Soulte", line=dict(color=C[3]))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title=xlab, yaxis_title="Soulte", template=DARK,
                      height=H, margin=dict(t=28,b=30,l=40,r=10),
                      showlegend=False)
    return fig

# Row 1: vs YTM
r1a, r1b = st.columns(2)
with r1a:
    st.caption("Spreads vs YTM")
    st.plotly_chart(spread_fig(ytm_g*100, pa_ytm, zs_ytm, ya_ytm, "YTM (%)"), use_container_width=True)
with r1b:
    st.caption("Soulte vs YTM")
    st.plotly_chart(soulte_fig(ytm_g*100, so_ytm, "YTM (%)"), use_container_width=True)

# Row 2: vs Risk-Free
r2a, r2b = st.columns(2)
with r2a:
    st.caption("Spreads vs Risk-Free Rate")
    st.plotly_chart(spread_fig(rf_g*100, pa_rf, zs_rf, ya_rf, "Risk-Free (%)"), use_container_width=True)
with r2b:
    st.caption("Soulte vs Risk-Free Rate")
    st.plotly_chart(soulte_fig(rf_g*100, so_rf, "Risk-Free (%)"), use_container_width=True)

# Row 3: vs Coupon & Maturity
r3a, r3b, r3c, r3d = st.columns(4)
with r3a:
    st.caption("Spreads vs Coupon")
    st.plotly_chart(spread_fig(cpn_g*100, pa_cpn, zs_cpn, ya_cpn, "Coupon (%)"), use_container_width=True)
with r3b:
    st.caption("Soulte vs Coupon")
    st.plotly_chart(soulte_fig(cpn_g*100, so_cpn, "Coupon (%)"), use_container_width=True)
with r3c:
    st.caption("Spreads vs Maturity")
    st.plotly_chart(spread_fig(mat_g, pa_mat, zs_mat, ya_mat, "Maturity (yr)"), use_container_width=True)
with r3d:
    st.caption("Soulte vs Maturity")
    st.plotly_chart(soulte_fig(mat_g, so_mat, "Maturity (yr)"), use_container_width=True)

# Row 4: Duration/DV01 + Leg PVs + Cashflows
r4a, r4b, r4c = st.columns(3)

with r4a:
    st.caption("Duration & DV01 vs YTM")
    durs, dv01s = [], []
    for y_ in ytm_g:
        res = metrics(coupon, maturity, y_, rf, face, freq)
        durs.append(res["dur"]); dv01s.append(res["dv01"])
    fig_d = make_subplots(specs=[[{"secondary_y": True}]])
    fig_d.add_trace(go.Scatter(x=ytm_g*100, y=durs,  name="Dur",  line=dict(color=C[4])), secondary_y=False)
    fig_d.add_trace(go.Scatter(x=ytm_g*100, y=dv01s, name="DV01", line=dict(color=C[5], dash="dash")), secondary_y=True)
    fig_d.update_layout(template=DARK, height=H, margin=dict(t=28,b=30,l=40,r=40),
                        legend=dict(orientation="h", y=1.12, x=0))
    fig_d.update_xaxes(title_text="YTM (%)")
    st.plotly_chart(fig_d, use_container_width=True)

with r4b:
    st.caption("Fixed PV / Float PV / Soulte vs YTM")
    fp, flp, so2 = [], [], []
    for y_ in ytm_g:
        res = metrics(coupon, maturity, y_, rf, face, freq)
        fp.append(res["fixed_pv"]); flp.append(res["float_pv"]); so2.append(res["soulte"])
    fig_l = go.Figure()
    fig_l.add_scatter(x=ytm_g*100, y=fp,  name="Fixed PV",  line=dict(color=C[0]))
    fig_l.add_scatter(x=ytm_g*100, y=flp, name="Float PV",  line=dict(color=C[1], dash="dash"))
    fig_l.add_scatter(x=ytm_g*100, y=so2, name="Soulte",    line=dict(color=C[3], dash="dot"))
    fig_l.add_vline(x=ytm*100, line_dash="dot", line_color="white")
    fig_l.update_layout(xaxis_title="YTM (%)", template=DARK, height=H,
                        margin=dict(t=28,b=30,l=40,r=10),
                        legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig_l, use_container_width=True)

with r4c:
    st.caption("Cashflow Schedule")
    n  = max(1, int(round(maturity * freq)))
    ts = np.arange(1, n + 1) / freq
    cpn_cf = np.full(n, coupon / freq * face)
    prin   = np.zeros(n); prin[-1] = face
    fig_cf = go.Figure()
    fig_cf.add_bar(x=ts, y=cpn_cf, name="Coupon",    marker_color=C[0])
    fig_cf.add_bar(x=ts, y=prin,   name="Principal", marker_color=C[1])
    fig_cf.update_layout(barmode="stack", xaxis_title="Time (yr)", template=DARK,
                         height=H, margin=dict(t=28,b=30,l=40,r=10),
                         legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig_cf, use_container_width=True)

# Row 5: 3D surface
st.markdown("---")
st.caption("Par ASW Surface — Maturity × YTM")

@st.cache_data(show_spinner=False)
def _surface(c, r_, F, freq_):
    mats = np.linspace(1, 15, 35)
    ytms = np.linspace(max(0.001, r_ - 0.04), r_ + 0.10, 35)
    Z = np.array([[par_asw(c, m, y_, r_, F, freq_) for y_ in ytms] for m in mats])
    return mats, ytms, Z

mats_s, ytms_s, Z_s = _surface(coupon, rf, face, freq)
fig_s = go.Figure(go.Surface(x=ytms_s*100, y=mats_s, z=Z_s, colorscale="Viridis"))
fig_s.update_layout(
    scene=dict(xaxis_title="YTM (%)", yaxis_title="Maturity (yr)", zaxis_title="Par ASW (bps)"),
    template=DARK, height=480, margin=dict(t=20,b=0,l=0,r=0),
)
st.plotly_chart(fig_s, use_container_width=True)
