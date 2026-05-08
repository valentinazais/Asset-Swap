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
# Compounding: annual (y-axis discount factors use annual spot rates)
# Coupons: paid at t = 1/freq, 2/freq, ..., T (maturity always receives principal)
# Stub: SHORT STUB — if T is not at coupon date, last coupon is accrued proportionally
# Settlement: t=0, no accrued interest adjustment

# ── MATH ──────────────────────────────────────────────────────────────────────

def discount_factors(rate: float, T: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate coupon dates and corresponding discount factors.
    
    Args:
        rate: annual spot rate (decimal, e.g., 0.05 for 5%)
        T: maturity in years
        freq: coupon frequency (1=annual, 2=semi, 4=quarterly)
    
    Returns:
        t: array of all cash flow dates [1/freq, 2/freq, ..., T]
        df: discount factors [exp(-rate * t_i)]
    """
    # All coupon dates < T
    n_coupons = int(np.floor(T * freq))
    t_coupons = np.array([k / freq for k in range(1, n_coupons + 1)])
    
    # Always include maturity date T
    t = np.append(t_coupons, T) if T > t_coupons[-1] + 1e-12 else t_coupons
    t = np.unique(np.round(t, 12))  # Remove duplicates at machine precision
    
    df = np.exp(-rate * t)
    return t, df


def cashflows(coupon: float, T: float, F: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build coupon and principal cash flows.
    
    Args:
        coupon: annual coupon rate (decimal, e.g., 0.05 for 5%)
        T: maturity in years
        F: face value (default 100)
        freq: coupon frequency
    
    Returns:
        t: cash flow dates
        cf: cash flow amounts [coupon, coupon, ..., coupon + principal]
    """
    t, _ = discount_factors(0, T, freq)
    n_coupons = int(np.floor(T * freq))
    coupon_period = 1 / freq
    
    cf = np.zeros(len(t))
    
    # All coupon dates before T get full coupon
    for i, date in enumerate(t):
        if date < T - 1e-12:
            cf[i] = coupon * coupon_period * F
    
    # Maturity date T: accrued coupon (if short stub) + principal
    last_coupon_date = n_coupons / freq
    if T > last_coupon_date + 1e-12:
        # Short stub: accrued coupon from last_coupon_date to T
        stub_duration = T - last_coupon_date  # in years
        stub_frac = stub_duration * freq      # fraction of coupon period [0, 1)
        cf[-1] += coupon * stub_frac * coupon_period * F
    else:
        # T is exactly at coupon date, full last coupon
        cf[-1] += coupon * coupon_period * F
    
    cf[-1] += F  # Principal at maturity
    
    return t, cf


def price_dirty(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Bond dirty price (includes accrued).
    
    Args:
        coupon: annual coupon rate
        T: maturity in years
        ytm: yield to maturity (annual, decimal)
        F: face value
        freq: coupon frequency
    
    Returns:
        Dirty price as percentage of par (e.g., 98.50)
    """
    t, cf = cashflows(coupon, T, F, freq)
    df = np.exp(-ytm * t)
    return float(np.dot(cf, df)) / F * 100.0


def annuity_factor(rate: float, T: float, freq: int) -> float:
    """
    Annuity factor = sum of discount factors at all coupon dates.
    
    Used for par ASW: represents PV of 1 bps coupon stream.
    """
    t, df = discount_factors(rate, T, freq)
    coupon_period = 1 / freq
    
    # Sum DF at all coupon dates (including T), weighted by period
    ann = np.sum(df) * coupon_period
    return float(ann)


def par_asw(coupon: float, T: float, ytm: float, rf_rate: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Par asset swap spread.
    
    Solves: PV(coupon + s_asw) = PV(ytm)
    Equivalently: s_asw = (ytm - rf) at par ASW equivalent.
    
    Formula:
        Par ASW = (ytm - rf_rate) / annuity_factor * 1e4 bps
    
    Standard interpretation: spread paid quarterly/semi-annually.
    """
    P_ytm = price_dirty(coupon, T, ytm, F, freq)
    ann = annuity_factor(rf_rate, T, freq)
    
    if ann < 1e-12:
        return 0.0
    
    # ASW converts bond yield premium to fixed spread
    # (c + s/1e4) discounted at rf equals bond price discounted at ytm
    s_asw = ((ytm - rf_rate) / (1 / ann)) * 1e4
    return float(s_asw)


def z_spread(coupon: float, T: float, ytm: float, rf_rate: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Z-spread (zero-volatility spread).
    
    Solves: PV_bond(z + rf_rate) = market_price
    where PV uses rf spot curve (here: flat) + z-spread.
    """
    P_market = price_dirty(coupon, T, ytm, F, freq)
    t, cf = cashflows(coupon, T, F, freq)
    
    def objective(z_bps):
        z = z_bps / 1e4
        df_spread = np.exp(-(rf_rate + z) * t)
        pv = np.dot(cf, df_spread)
        return pv - P_market
    
    try:
        z_result = brentq(objective, -0.05, 0.10, xtol=1e-10)
        return float(z_result * 1e4)
    except Exception:
        return float("nan")


def yield_asw(ytm: float, rf_rate: float) -> float:
    """
    Yield ASW: simple ytm - rf spread (does not account for annuity adjustment).
    """
    return float((ytm - rf_rate) * 1e4)


def soulte(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Initial payment to make asset swap par.
    
    soulte = market_price - 100
    (negative = buyer receives discount, positive = buyer pays premium)
    """
    P = price_dirty(coupon, T, ytm, F, freq)
    return float(P - F)


def modified_duration(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    Modified duration: -(1/P) * dP/dy, where y is annual yield.
    
    Calculated via Macaulay duration scaled by annual compounding.
    """
    t, cf = cashflows(coupon, T, F, freq)
    df = np.exp(-ytm * t)
    P = np.dot(cf, df)
    
    if P < 1e-12:
        return 0.0
    
    # Macaulay duration (in years)
    D_mac = np.dot(t * cf, df) / P
    
    # Modified = Macaulay / (1 + y/freq)
    # With annual compounding: Modified = Macaulay / (1 + y)
    D_mod = D_mac / (1 + ytm)
    return float(D_mod)


def dv01(coupon: float, T: float, ytm: float, F: float = 100.0, freq: int = 2) -> float:
    """
    DV01: price change per 1 basis point change in yield.
    
    DV01 = -Modified_Duration * Price * 0.0001
    """
    P = price_dirty(coupon, T, ytm, F, freq)
    D = modified_duration(coupon, T, ytm, F, freq)
    return float(D * P / 100.0 * 0.0001)

# ── CONTROLS ──────────────────────────────────────────────────────────────────

st.title("Asset Swap Pricer")

c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 1.2, 1.2, 1.4, 1.2, 1.2, 0.8])
with c1:
    face   = st.number_input("Face",          min_value=10.0,  max_value=10000.0, value=100.0, step=10.0)
with c2:
    coupon = st.number_input("Coupon (%)",    min_value=0.01,  max_value=30.0,    value=5.0,   step=0.5) / 100.0
with c3:
    mat    = st.number_input("Maturity (yr)", min_value=0.5,   max_value=30.0,    value=5.0,   step=0.5)
with c4:
    freq_label = st.selectbox("Freq", ["Annual", "Semi", "Quarterly"], index=1)
    freq = {"Annual": 1, "Semi": 2, "Quarterly": 4}[freq_label]
with c5:
    ytm    = st.number_input("YTM (%)",       min_value=0.01,  max_value=30.0,    value=5.5,   step=0.5) / 100.0
with c6:
    rf     = st.number_input("Risk-Free (%)", min_value=0.01,  max_value=25.0,    value=3.0,   step=0.5) / 100.0
with c7:
    if st.button("Reset"):
        st.rerun()

# ── METRICS ───────────────────────────────────────────────────────────────────

P_dirty = price_dirty(coupon, mat, ytm, face, freq)
P_soulte = soulte(coupon, mat, ytm, face, freq)
P_asw = par_asw(coupon, mat, ytm, rf, face, freq)
P_zspread = z_spread(coupon, mat, ytm, rf, face, freq)
P_yield_asw = yield_asw(ytm, rf)
D_mod = modified_duration(coupon, mat, ytm, face, freq)
P_dv01 = dv01(coupon, mat, ytm, face, freq)
Ann = annuity_factor(rf, mat, freq)

m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
with m1: st.metric("Dirty Price", f"{P_dirty:.4f}")
with m2: st.metric("Soulte", f"{P_soulte:.4f}")
with m3: st.metric("Par ASW (bps)", f"{P_asw:.2f}")
with m4: st.metric("Z-Spread (bps)", f"{P_zspread:.2f}")
with m5: st.metric("Yield ASW (bps)", f"{P_yield_asw:.2f}")
with m6: st.metric("Mod. Duration", f"{D_mod:.4f}")
with m7: st.metric("DV01", f"{P_dv01:.6f}")
with m8: st.metric("Annuity", f"{Ann:.6f}")

# ── SENSITIVITY GRIDS ─────────────────────────────────────────────────────────

ytm_grid = np.linspace(max(0.001, rf - 0.02), rf + 0.12, 50)
mat_grid = np.linspace(0.5, 15.0, 50)
coupon_grid = np.array([0.01, 0.03, 0.05, 0.08])

# Graph 1: Spread Conventions vs YTM
pasw_g = np.array([par_asw(coupon, mat, y, rf, face, freq) for y in ytm_grid])
zspr_g = np.array([z_spread(coupon, mat, y, rf, face, freq) for y in ytm_grid])
yasw_g = np.array([yield_asw(y, rf) for y in ytm_grid])

fig1 = go.Figure()
fig1.add_scatter(x=ytm_grid*100, y=pasw_g, name="Par ASW", 
                 line=dict(color=CA, width=2))
fig1.add_scatter(x=ytm_grid*100, y=zspr_g, name="Z-Spread", 
                 line=dict(color=CB, width=2, dash="dash"))
fig1.add_scatter(x=ytm_grid*100, y=yasw_g, name="Yield ASW", 
                 line=dict(color=CC, width=2, dash="dot"))
fig1.add_vline(x=ytm*100, line_dash="solid", line_color="white", line_width=1)
fig1.update_layout(
    template=DARK, height=H, margin=MARGIN,
    title=dict(text="Spread Conventions vs YTM", font=dict(size=15)),
    xaxis=dict(title="YTM (%)", tickfont=dict(size=11)),
    yaxis=dict(title="Spread (bps)", tickfont=dict(size=11)),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    hovermode="x unified"
)

# Graph 2: Soulte vs YTM
soul_g = np.array([soulte(coupon, mat, y, face, freq) for y in ytm_grid])

fig2 = go.Figure()
fig2.add_scatter(x=ytm_grid*100, y=soul_g, name="Soulte", 
                 line=dict(color=CB, width=2.5), fill="tozeroy", fillcolor="rgba(239,35,60,0.2)")
fig2.add_hline(y=0, line_dash="dot", line_color="white", line_width=1)
fig2.add_vline(x=ytm*100, line_dash="solid", line_color="white", line_width=1)
fig2.update_layout(
    template=DARK, height=H, margin=MARGIN,
    title=dict(text="Soulte vs YTM (zero at par)", font=dict(size=15)),
    xaxis=dict(title="YTM (%)", tickfont=dict(size=11)),
    yaxis=dict(title="Soulte (price)", tickfont=dict(size=11)),
    hovermode="x unified"
)

# Graph 3: Par ASW Term Structure by Coupon
fig3 = go.Figure()
for cp in coupon_grid:
    ts_vals = np.array([par_asw(cp, m, ytm, rf, face, freq) for m in mat_grid])
    colors = [CA, CB, "#f4a261", "#e76f51"]
    col = colors[list(coupon_grid).index(cp)]
    fig3.add_scatter(x=mat_grid, y=ts_vals, name=f"c={cp*100:.1f}%", 
                     line=dict(color=col, width=2))
fig3.add_vline(x=mat, line_dash="solid", line_color="white", line_width=1)
fig3.update_layout(
    template=DARK, height=H, margin=MARGIN,
    title=dict(text="Par ASW Term Structure — by Coupon", font=dict(size=15)),
    xaxis=dict(title="Maturity (yr)", tickfont=dict(size=11)),
    yaxis=dict(title="Par ASW (bps)", tickfont=dict(size=11)),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    hovermode="x unified"
)

# Graph 4: Gap (Par ASW − Z-Spread) + Soulte
gap_g = pasw_g - zspr_g

fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_scatter(x=ytm_grid*100, y=gap_g, name="Par ASW − Z-Spread (bps)",
                 line=dict(color=CA, width=2), secondary_y=False)
fig4.add_scatter(x=ytm_grid*100, y=soul_g, name="Soulte",
                 line=dict(color=CB, width=2, dash="dash"), secondary_y=True)
fig4.add_hline(y=0, line_dash="dot", line_color="white", line_width=1, secondary_y=False)
fig4.add_vline(x=ytm*100, line_dash="solid", line_color="white", line_width=1)
fig4.update_layout(
    template=DARK, height=H, margin=MARGIN,
    title=dict(text="Par ASW vs Z-Spread Gap & Soulte", font=dict(size=15)),
    xaxis=dict(title="YTM (%)", tickfont=dict(size=11)),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    hovermode="x unified"
)
fig4.update_yaxes(title_text="Gap (bps)", secondary_y=False, tickfont=dict(size=11))
fig4.update_yaxes(title_text="Soulte", secondary_y=True, tickfont=dict(size=11))

# Graph 5: Duration & DV01 vs YTM
md_g = np.array([modified_duration(coupon, mat, y, face, freq) for y in ytm_grid])
dv_g = np.array([dv01(coupon, mat, y, face, freq) for y in ytm_grid])

fig5 = make_subplots(specs=[[{"secondary_y": True}]])
fig5.add_scatter(x=ytm_grid*100, y=md_g, name="Mod. Duration",
                 line=dict(color=CA, width=2), secondary_y=False)
fig5.add_scatter(x=ytm_grid*100, y=dv_g, name="DV01",
                 line=dict(color=CC, width=2, dash="dash"), secondary_y=True)
fig5.add_vline(x=ytm*100, line_dash="solid", line_color="white", line_width=1)
fig5.update_layout(
    template=DARK, height=H, margin=MARGIN,
    title=dict(text="Duration & DV01 vs YTM", font=dict(size=15)),
    xaxis=dict(title="YTM (%)", tickfont=dict(size=11)),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    hovermode="x unified"
)
fig5.update_yaxes(title_text="Mod. Duration", secondary_y=False, tickfont=dict(size=11))
fig5.update_yaxes(title_text="DV01", secondary_y=True, tickfont=dict(size=11))

# Graph 6: Par ASW Surface
@st.cache_data(show_spinner=False)
def _surface(c, r_, F, freq_):
    mats_ = np.linspace(1, 15, 40)
    ytms_ = np.linspace(max(0.002, r_ - 0.02), r_ + 0.12, 40)
    Z = np.array([[par_asw(c, m_, y_, r_, F, freq_) for y_ in ytms_] for m_ in mats_])
    return mats_, ytms_, Z

mats_s, ytms_s, Z_s = _surface(coupon, rf, face, freq)

fig6 = go.Figure(go.Surface(
    x=ytms_s*100, y=mats_s, z=Z_s, 
    colorscale="Viridis",
    colorbar=dict(title="Par ASW (bps)", thickness=15, len=0.7)
))
fig6.update_layout(
    scene=dict(
        xaxis=dict(title="YTM (%)", tickfont=dict(size=10), title_font=dict(size=12)),
        yaxis=dict(title="Maturity (yr)", tickfont=dict(size=10), title_font=dict(size=12)),
        zaxis=dict(title="Par ASW (bps)", tickfont=dict(size=10), title_font=dict(size=12)),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    title=dict(text="Par ASW Surface — Maturity × YTM", font=dict(size=15)),
    template=DARK, 
    height=H, 
    margin=dict(t=50, b=10, l=10, r=10),
    hovermode="closest"
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
