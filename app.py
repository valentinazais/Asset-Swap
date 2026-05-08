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

# ── MATH ──────────────────────────────────────────────────────────────────────

def disc(r, T, freq):
    """
    Retourne (t, df) où :
      t  : dates des cash flows — la dernière vaut exactement T
      df : facteurs d'actualisation exp(-r * t)
    Correction : t[-1] = T exact, élimine la discontinuité due à l'arrondi.
    """
    n     = max(1, int(round(T * freq)))
    t     = np.arange(1, n + 1) / freq
    t[-1] = T                                        # FIX 1 : dernière date = T exact
    return t, np.exp(-r * t)

def dirty(c, T, y, F=100.0, freq=2):
    t, df = disc(y, T, freq)
    cf    = np.full(len(t), c / freq * F)
    cf[-1] += F
    return float(np.dot(cf, df))

def par_asw(c, T, y, r, F=100.0, freq=2):
    t, df_rf = disc(r, T, freq)
    ann = float(np.sum(df_rf / freq))
    ZT  = float(df_rf[-1])                           # FIX 2 : cohérent avec disc()
    P   = dirty(c, T, y, F, freq) / F
    return float((c * ann + ZT - P) / ann * 1e4) if ann > 1e-12 else 0.0

def z_spread(c, T, y, r, F=100.0, freq=2):
    P        = dirty(c, T, y, F, freq)
    t, _     = disc(r, T, freq)                      # FIX 3 : hors closure
    cf       = np.full(len(t), c / freq * F)
    cf[-1]  += F
    def pv(s):
        df = np.exp(-(r + s) * t)
        return float(np.dot(cf, df)) - P
    try:
        return brentq(pv, -0.5, 5.0, xtol=1e-10) * 1e4
    except Exception:
        return float("nan")

def yield_asw(c, T, y, r, F=100.0, freq=2):
    return (y - r) * 1e4

def soulte(c, T, y, F=100.0, freq=2):
    return dirty(c, T, y, F, freq) - F

def mod_dur(c, T, y, F=100.0, freq=2):
    t, df = disc(y, T, freq)
    cf    = np.full(len(t), c / freq * F)
    cf[-1] += F
    P     = float(np.dot(cf, df))
    D_mac = float(np.dot(t * cf, df)) / P
    return D_mac / (1 + y / freq)

def dv01_fn(c, T, y, F=100.0, freq=2):
    return (dirty(c, T, y - 5e-5, F, freq) - dirty(c, T, y + 5e-5, F, freq)) / 2.0

# ── CONTROLS ──────────────────────────────────────────────────────────────────

st.title("Asset Swap Pricer")

c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 1.2, 1.2, 1.4, 1.2, 1.2, 0.8])
with c1:
    face   = st.number_input("Face",          min_value=10.0,  max_value=10000.0, value=100.0, step=10.0)
with c2:
    coupon = st.number_input("Coupon",        min_value=0.001, max_value=0.30,    value=0.05,  step=0.005, format="%.3f")
with c3:
    mat    = st.number_input("Maturity (yr)", min_value=0.5,   max_value=30.0,    value=5.0,   step=0.5)
with c4:
    freq_label = st.selectbox("Freq", ["Annual", "Semi", "Quarterly"], index=1)
    freq = {"Annual": 1, "Semi": 2, "Quarterly": 4}[freq_label]
with c5:
    ytm    = st.number_input("YTM",           min_value=0.001, max_value=0.30,    value=0.055, step=0.005, format="%.3f")
with c6:
    rf     = st.number_input("Risk-Free",     min_value=0.001, max_value=0.25,    value=0.03,  step=0.005, format="%.3f")
with c7:
    if st.button("Reset"):
        st.rerun()

# ── METRICS ───────────────────────────────────────────────────────────────────

dp   = dirty(coupon, mat, ytm, face, freq)
soul = soulte(coupon, mat, ytm, face, freq)
pasw = par_asw(coupon, mat, ytm, rf, face, freq)
zspr = z_spread(coupon, mat, ytm, rf, face, freq)
yasw = yield_asw(coupon, mat, ytm, rf, face, freq)
md   = mod_dur(coupon, mat, ytm, face, freq)
dv   = dv01_fn(coupon, mat, ytm, face, freq)
t_rf, df_rf = disc(rf, mat, freq)
ann  = float(np.sum(df_rf / freq))

m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
m1.metric("Dirty Price",     f"{dp:.4f}")
m2.metric("Soulte",          f"{soul:.4f}")
m3.metric("Par ASW (bps)",   f"{pasw:.2f}")
m4.metric("Z-Spread (bps)",  f"{zspr:.2f}")
m5.metric("Yield ASW (bps)", f"{yasw:.2f}")
m6.metric("Mod. Duration",   f"{md:.4f}")
m7.metric("DV01",            f"{dv:.6f}")
m8.metric("Annuity",         f"{ann:.6f}")

st.divider()

# ── GRID PARAMS ───────────────────────────────────────────────────────────────

ytm_grid = np.linspace(max(0.002, rf - 0.02), rf + 0.14, 120)
mat_grid = np.linspace(0.5, 20.0, 400)          # résolution doublée

LEGEND = dict(orientation="h", y=1.06, x=0, font=dict(size=13))
AXIS   = dict(tickfont=dict(size=12), title_font=dict(size=13))

def vline(fig, x, row=None, col=None):
    kw = dict(x0=x, x1=x, line=dict(color="white", width=1, dash="dot"),
              xref="x", yref="paper", y0=0, y1=1, type="line")
    if row:
        fig.add_shape(**kw, row=row, col=col)
    else:
        fig.add_shape(**kw)

# ── GRAPH 1 : Spread Conventions vs YTM ──────────────────────────────────────

pasw_g = [par_asw(coupon, mat, y, rf, face, freq)   for y in ytm_grid]
zspr_g = [z_spread(coupon, mat, y, rf, face, freq)  for y in ytm_grid]
yasw_g = [yield_asw(coupon, mat, y, rf, face, freq) for y in ytm_grid]

fig1 = go.Figure()
fig1.add_scatter(x=ytm_grid*100, y=pasw_g, name="Par ASW",   line=dict(color=CA, width=2))
fig1.add_scatter(x=ytm_grid*100, y=zspr_g, name="Z-Spread",  line=dict(color=CB, width=2, dash="dash"))
fig1.add_scatter(x=ytm_grid*100, y=yasw_g, name="Yield ASW", line=dict(color=CC, width=2, dash="dot"))
vline(fig1, ytm*100)
fig1.update_layout(template=DARK, height=H, margin=MARGIN,
                   title=dict(text="Spread Conventions vs YTM", font=dict(size=15)),
                   xaxis=dict(title="YTM (%)", **AXIS),
                   yaxis=dict(title="bps", **AXIS),
                   legend=LEGEND)

# ── GRAPH 2 : Soulte vs YTM ───────────────────────────────────────────────────

soul_g = [soulte(coupon, mat, y, face, freq) for y in ytm_grid]

fig2 = go.Figure()
fig2.add_scatter(x=ytm_grid*100, y=soul_g, line=dict(color=CB, width=2.5), showlegend=False)
fig2.add_hline(y=0, line_dash="dot", line_color="white", line_width=1)
vline(fig2, ytm*100)
fig2.add_vline(x=coupon*100, line_dash="dash", line_color="#aaa", line_width=1,
               annotation_text="YTM = Coupon", annotation_font_size=12,
               annotation_position="top right")
fig2.update_layout(template=DARK, height=H, margin=MARGIN,
                   title=dict(text="Soulte vs YTM  (zero at par)", font=dict(size=15)),
                   xaxis=dict(title="YTM (%)", **AXIS),
                   yaxis=dict(title="Soulte", **AXIS))

# ── GRAPH 3 : Par ASW Term Structure by Coupon ────────────────────────────────

coupons_ts = [0.01, 0.03, 0.05, 0.08]
colors_ts  = [CA, "#48cae4", CC, CB]

fig3 = go.Figure()
for cp, col in zip(coupons_ts, colors_ts):
    ts_vals = [par_asw(cp, m, ytm, rf, face, freq) for m in mat_grid]
    fig3.add_scatter(x=mat_grid, y=ts_vals,
                     name=f"c={cp:.1%}", line=dict(color=col, width=2))
vline(fig3, mat)
fig3.update_layout(template=DARK, height=H, margin=MARGIN,
                   title=dict(text="Par ASW Term Structure — by Coupon", font=dict(size=15)),
                   xaxis=dict(title="Maturity (yr)", **AXIS),
                   yaxis=dict(title="Par ASW (bps)", **AXIS),
                   legend=LEGEND)

# ── GRAPH 4 : Gap (Par ASW − Z-Spread) + Soulte ──────────────────────────────

gap_g = [p - z for p, z in zip(pasw_g, zspr_g)]

fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_scatter(x=ytm_grid*100, y=gap_g,  name="Par ASW − Z-Spread (bps)",
                 line=dict(color=CA, width=2), secondary_y=False)
fig4.add_scatter(x=ytm_grid*100, y=soul_g, name="Soulte",
                 line=dict(color=CB, width=2, dash="dash"), secondary_y=True)
fig4.add_hline(y=0, line_dash="dot", line_color="white", line_width=1, secondary_y=False)
vline(fig4, ytm*100, row=1, col=1)
fig4.update_layout(template=DARK, height=H, margin=MARGIN,
                   title=dict(text="Par ASW vs Z-Spread Gap & Soulte", font=dict(size=15)),
                   xaxis=dict(title="YTM (%)", **AXIS),
                   legend=LEGEND)
fig4.update_yaxes(title_text="Gap (bps)", secondary_y=False, **AXIS)
fig4.update_yaxes(title_text="Soulte",    secondary_y=True,  **AXIS)

# ── GRAPH 5 : Duration & DV01 vs YTM ─────────────────────────────────────────

md_g = [mod_dur(coupon, mat, y, face, freq) for y in ytm_grid]
dv_g = [dv01_fn(coupon, mat, y, face, freq) for y in ytm_grid]

fig5 = make_subplots(specs=[[{"secondary_y": True}]])
fig5.add_scatter(x=ytm_grid*100, y=md_g, name="Mod. Duration",
                 line=dict(color=CA, width=2), secondary_y=False)
fig5.add_scatter(x=ytm_grid*100, y=dv_g, name="DV01",
                 line=dict(color=CC, width=2, dash="dash"), secondary_y=True)
vline(fig5, ytm*100, row=1, col=1)
fig5.update_layout(template=DARK, height=H, margin=MARGIN,
                   title=dict(text="Duration & DV01 vs YTM", font=dict(size=15)),
                   xaxis=dict(title="YTM (%)", **AXIS),
                   legend=LEGEND)
fig5.update_yaxes(title_text="Mod. Duration", secondary_y=False, **AXIS)
fig5.update_yaxes(title_text="DV01",          secondary_y=True,  **AXIS)

# ── GRAPH 6 : Par ASW Surface ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _surface(c, r_, F, freq_):
    mats_ = np.linspace(1, 15, 35)
    ytms_ = np.linspace(max(0.002, r_ - 0.02), r_ + 0.12, 35)
    Z     = np.array([[par_asw(c, m_, y_, r_, F, freq_) for y_ in ytms_] for m_ in mats_])
    return mats_, ytms_, Z

mats_s, ytms_s, Z_s = _surface(coupon, rf, face, freq)
fig6 = go.Figure(go.Surface(x=ytms_s*100, y=mats_s, z=Z_s, colorscale="Viridis"))
fig6.update_layout(
    scene=dict(
        xaxis=dict(title="YTM (%)",       tickfont=dict(size=11), title_font=dict(size=13)),
        yaxis=dict(title="Maturity (yr)", tickfont=dict(size=11), title_font=dict(size=13)),
        zaxis=dict(title="Par ASW (bps)", tickfont=dict(size=11), title_font=dict(size=13)),
    ),
    title=dict(text="Par ASW Surface — Maturity × YTM", font=dict(size=15)),
    template=DARK, height=H, margin=dict(t=50, b=10, l=10, r=10),
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
