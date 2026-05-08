# run: streamlit run app.py

from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

st.set_page_config(page_title="Asset Swap", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:0.6rem;padding-bottom:0rem;}
div[data-testid="metric-container"]{background:#1a1a2e;border-radius:5px;padding:4px 8px;}
div[data-testid="metric-container"] label{font-size:0.7rem !important;}
div[data-testid="metric-container"] div{font-size:0.95rem !important;}
</style>
""", unsafe_allow_html=True)

# ── MATH ──────────────────────────────────────────────────────────────────────

def discount_factors(r, T, freq):
    n = max(1, int(round(T * freq)))
    t = np.arange(1, n + 1) / freq
    return t, np.exp(-r * t)

def bond_price(c, T, y, F=100.0, freq=2):
    t, df = discount_factors(y, T, freq)
    cf = np.full(len(t), c / freq * F)
    cf[-1] += F
    return float(np.dot(cf, df))

def par_asw(c, T, y, r, F=100.0, freq=2):
    t, df_rf = discount_factors(r, T, freq)
    ann = float(np.sum(df_rf)) / freq
    ZT  = np.exp(-r * T)
    P   = bond_price(c, T, y, F, freq) / F
    return float((c * ann + ZT - P) / ann) * 1e4 if ann > 1e-12 else 0.0

def z_spread(c, T, y, r, F=100.0, freq=2):
    dp = bond_price(c, T, y, F, freq)
    def pv_diff(z):
        t, _ = discount_factors(r, T, freq)
        df = np.exp(-(r + z) * t)
        cf = np.full(len(t), c / freq * F); cf[-1] += F
        return float(np.dot(cf, df)) - dp
    try:
        return brentq(pv_diff, -0.5, 5.0, xtol=1e-12) * 1e4
    except Exception:
        return float("nan")

def mod_duration(c, T, y, F=100.0, freq=2):
    t, df = discount_factors(y, T, freq)
    cf = np.full(len(t), c / freq * F); cf[-1] += F
    P = float(np.dot(cf, df))
    return float(np.dot(t * cf, df)) / P / (1 + y / freq)

def all_metrics(c, T, y, r, F, freq):
    P   = bond_price(c, T, y, F, freq)
    sol = P - F
    pa  = par_asw(c, T, y, r, F, freq)
    zs  = z_spread(c, T, y, r, F, freq)
    ya  = (y - r) * 1e4
    md  = mod_duration(c, T, y, F, freq)
    dv01 = P * md / 1e4
    t, df_rf = discount_factors(r, T, freq)
    ann = float(np.sum(df_rf)) / freq
    return dict(P=P, sol=sol, pa=pa, zs=zs, ya=ya, md=md, dv01=dv01, ann=ann)

# ── CONTROLS ──────────────────────────────────────────────────────────────────

st.markdown("### Asset Swap Pricer")

if "rc" not in st.session_state:
    st.session_state.rc = 0
rc = st.session_state.rc

cols = st.columns([1.2, 1.2, 1.2, 1.0, 1.2, 1.2, 0.5])
face   = cols[0].number_input("Face",          value=100.0, step=10.0,  format="%.0f",  key=f"F_{rc}")
coupon = cols[1].number_input("Coupon",        value=0.050, step=0.005, format="%.3f",  min_value=0.001, max_value=0.25,  key=f"c_{rc}")
mat    = cols[2].number_input("Maturity (yr)", value=5.0,   step=0.5,   format="%.1f",  min_value=0.5,   max_value=30.0,  key=f"T_{rc}")
freq   = cols[3].selectbox("Freq", [1,2,4], index=1, format_func=lambda x:{1:"Ann",2:"Semi",4:"Qtr"}[x], key=f"fr_{rc}")
ytm    = cols[4].number_input("YTM",           value=0.055, step=0.005, format="%.3f",  min_value=0.001, max_value=0.30,  key=f"y_{rc}")
rf     = cols[5].number_input("Risk-Free",     value=0.030, step=0.005, format="%.3f",  min_value=0.001, max_value=0.20,  key=f"r_{rc}")
with cols[6]:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻"):
        st.session_state.rc += 1
        st.rerun()

m = all_metrics(coupon, mat, ytm, rf, face, freq)

mc = st.columns(8)
mc[0].metric("Dirty Price",     f"{m['P']:.3f}")
mc[1].metric("Soulte",          f"{m['sol']:+.3f}")
mc[2].metric("Par ASW (bps)",   f"{m['pa']:.2f}")
mc[3].metric("Z-Spread (bps)",  f"{m['zs']:.2f}")
mc[4].metric("Yield ASW (bps)", f"{m['ya']:.2f}")
mc[5].metric("Mod. Duration",   f"{m['md']:.4f}")
mc[6].metric("DV01",            f"{m['dv01']:.5f}")
mc[7].metric("Annuity",         f"{m['ann']:.4f}")

st.divider()

# ── SWEEPS ────────────────────────────────────────────────────────────────────

N = 200
DARK   = "plotly_dark"
H      = 300
MARGIN = dict(t=30, b=36, l=44, r=12)
CA, CB, CC, CD = "#00b4d8", "#f77f00", "#06d6a0", "#e63946"

def vline(fig, x):
    fig.add_vline(x=x, line_dash="dot", line_color="rgba(255,255,255,0.4)", line_width=1)

# ── GRAPH 1 : Spread decomposition vs YTM ─────────────────────────────────────
# Shows WHY par ASW ≠ yield ASW ≠ z-spread and how they diverge

ytm_g = np.linspace(max(0.002, ytm - 0.07), ytm + 0.10, N)
pa_y, zs_y, ya_y = [], [], []
for y_ in ytm_g:
    res = all_metrics(coupon, mat, y_, rf, face, freq)
    pa_y.append(res["pa"]); zs_y.append(res["zs"]); ya_y.append(res["ya"])

fig1 = go.Figure()
fig1.add_scatter(x=ytm_g*100, y=pa_y, name="Par ASW",   line=dict(color=CA, width=2))
fig1.add_scatter(x=ytm_g*100, y=zs_y, name="Z-Spread",  line=dict(color=CB, width=2, dash="dash"))
fig1.add_scatter(x=ytm_g*100, y=ya_y, name="Yield ASW", line=dict(color=CC, width=2, dash="dot"))
vline(fig1, ytm*100)
fig1.update_layout(template=DARK, height=H, margin=MARGIN,
                   title="1 · Spread Conventions vs YTM",
                   xaxis_title="YTM (%)", yaxis_title="bps",
                   legend=dict(orientation="h", y=1.18, x=0, font_size=11))

# ── GRAPH 2 : Soulte vs YTM — core mechanism ──────────────────────────────────
# The soulte crosses zero at par (ytm = coupon): fundamental ASW concept

ytm_g2 = np.linspace(max(0.002, coupon - 0.08), coupon + 0.08, N)
so_y2  = [bond_price(coupon, mat, y_, face, freq) - face for y_ in ytm_g2]

fig2 = go.Figure()
fig2.add_scatter(x=ytm_g2*100, y=so_y2, line=dict(color=CD, width=2.5), name="Soulte")
fig2.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
fig2.add_vline(x=coupon*100, line_dash="dot", line_color="rgba(255,255,255,0.4)", line_width=1,
               annotation_text="YTM = Coupon", annotation_position="top right",
               annotation_font_color="white", annotation_font_size=10)
fig2.update_layout(template=DARK, height=H, margin=MARGIN,
                   title="2 · Soulte vs YTM  (zero at par)",
                   xaxis_title="YTM (%)", yaxis_title="Soulte",
                   showlegend=False)

# ── GRAPH 3 : Par ASW vs Maturity for different coupons ───────────────────────
# Shows term structure sensitivity and coupon effect

mat_g = np.linspace(0.5, 20.0, N)
coupons_test = [rf - 0.02, rf, coupon, coupon + 0.03]
colors_coup  = [CA, CC, CD, CB]

fig3 = go.Figure()
for cp, col in zip(coupons_test, colors_coup):
    if cp < 0.001:
        continue
    pa_m = [par_asw(cp, m_, ytm, rf, face, freq) for m_ in mat_g]
    label = f"c={cp*100:.1f}%"
    fig3.add_scatter(x=mat_g, y=pa_m, name=label, line=dict(color=col, width=2))
vline(fig3, mat)
fig3.update_layout(template=DARK, height=H, margin=MARGIN,
                   title="3 · Par ASW Term Structure — by Coupon",
                   xaxis_title="Maturity (yr)", yaxis_title="Par ASW (bps)",
                   legend=dict(orientation="h", y=1.18, x=0, font_size=11))

# ── GRAPH 4 : Par ASW vs Z-Spread gap vs Soulte ───────────────────────────────
# The gap (par ASW - z-spread) is driven by the soulte — key analytical insight

ytm_g4  = np.linspace(max(0.002, ytm - 0.07), ytm + 0.10, N)
gap4, sol4 = [], []
for y_ in ytm_g4:
    res = all_metrics(coupon, mat, y_, rf, face, freq)
    gap4.append(res["pa"] - res["zs"])
    sol4.append(res["sol"])

fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_trace(go.Scatter(x=ytm_g4*100, y=gap4, name="Par ASW − Z-Spread (bps)",
                           line=dict(color=CA, width=2)), secondary_y=False)
fig4.add_trace(go.Scatter(x=ytm_g4*100, y=sol4, name="Soulte",
                           line=dict(color=CD, width=2, dash="dash")), secondary_y=True)
fig4.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
vline(fig4, ytm*100)
fig4.update_layout(template=DARK, height=H, margin=MARGIN,
                   title="4 · Par ASW vs Z-Spread Gap ↔ Soulte",
                   legend=dict(orientation="h", y=1.18, x=0, font_size=11))
fig4.update_xaxes(title_text="YTM (%)")
fig4.update_yaxes(title_text="Gap (bps)",  secondary_y=False)
fig4.update_yaxes(title_text="Soulte",     secondary_y=True, showgrid=False)

# ── GRAPH 5 : Duration & DV01 vs YTM ─────────────────────────────────────────
# Interest rate sensitivity of the ASW position

durs, dv01s = [], []
for y_ in ytm_g:
    res = all_metrics(coupon, mat, y_, rf, face, freq)
    durs.append(res["md"]); dv01s.append(res["dv01"])

fig5 = make_subplots(specs=[[{"secondary_y": True}]])
fig5.add_trace(go.Scatter(x=ytm_g*100, y=durs,  name="Mod. Duration",
                           line=dict(color=CA, width=2)), secondary_y=False)
fig5.add_trace(go.Scatter(x=ytm_g*100, y=dv01s, name="DV01",
                           line=dict(color=CB, width=2, dash="dash")), secondary_y=True)
vline(fig5, ytm*100)
fig5.update_layout(template=DARK, height=H, margin=MARGIN,
                   title="5 · Duration & DV01 vs YTM",
                   legend=dict(orientation="h", y=1.18, x=0, font_size=11))
fig5.update_xaxes(title_text="YTM (%)")
fig5.update_yaxes(title_text="Mod. Duration", secondary_y=False)
fig5.update_yaxes(title_text="DV01",          secondary_y=True, showgrid=False)

# ── GRAPH 6 : Par ASW surface Maturity × YTM ─────────────────────────────────
# Full 2D view of how spread varies across the two most important dimensions

@st.cache_data(show_spinner=False)
def _surface(c, r_, F, freq_):
    mats = np.linspace(0.5, 15, 45)
    ytms = np.linspace(max(0.002, r_ - 0.02), r_ + 0.12, 45)
    Z = np.array([[par_asw(c, m_, y_, r_, F, freq_) for y_ in ytms] for m_ in mats])
    return mats, ytms, Z

mats_s, ytms_s, Z_s = _surface(coupon, rf, face, freq)
fig6 = go.Figure(go.Surface(
    x=ytms_s*100, y=mats_s, z=Z_s,
    colorscale="Viridis",
    contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
))
fig6.update_layout(
    title="6 · Par ASW Surface — Maturity × YTM",
    scene=dict(xaxis_title="YTM (%)", yaxis_title="Maturity (yr)", zaxis_title="Par ASW (bps)"),
    template=DARK, height=H + 50, margin=dict(t=30, b=0, l=0, r=0),
)

# ── LAYOUT 2×3 ────────────────────────────────────────────────────────────────

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1: st.plotly_chart(fig1, use_container_width=True)
with r1c2: st.plotly_chart(fig2, use_container_width=True)
with r1c3: st.plotly_chart(fig3, use_container_width=True)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1: st.plotly_chart(fig4, use_container_width=True)
with r2c2: st.plotly_chart(fig5, use_container_width=True)
with r2c3: st.plotly_chart(fig6, use_container_width=True)
