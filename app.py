import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricing import (asw_spread, bond_dirty_price, build_curve, cashflow_table,
                     forward_rate, bond_schedule)

st.set_page_config(page_title="Asset Swap Pricer", layout="wide")
st.title("Asset Swap Pricer")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Bond")
coupon_rate = st.sidebar.number_input("Coupon (%)", value=4.0, step=0.1) / 100
maturity = st.sidebar.number_input("Maturité (années)", value=5.0, step=0.5)
fix_freq = st.sidebar.selectbox("Fréquence coupon (par an)", [1, 2, 4], index=0)

price_mode = st.sidebar.radio("Input prix", ["YTM", "Dirty price direct"])
if price_mode == "YTM":
    ytm = st.sidebar.number_input("YTM (%)", value=3.5, step=0.1) / 100
    dirty_price = bond_dirty_price(coupon_rate, maturity, ytm, fix_freq)
else:
    dirty_price = st.sidebar.number_input("Dirty price", value=102.0, step=0.1)

st.sidebar.header("Jambe flottante")
flt_freq = st.sidebar.selectbox("Fréquence Libor (par an)", [2, 4, 12], index=1)

st.sidebar.header("Courbe zéro coupon")
default_curve = pd.DataFrame({
    "tenor": [1.0, 2.0, 5.0, 10.0],
    "rate (%)": [2.5, 2.8, 3.2, 3.5],
})
curve_df = st.sidebar.data_editor(default_curve, num_rows="dynamic")
tenors = curve_df["tenor"].tolist()
rates = (curve_df["rate (%)"] / 100).tolist()
DF = build_curve(tenors, rates)

# ---------------- CALCUL ----------------
spread, details = asw_spread(coupon_rate, maturity, dirty_price,
                              fix_freq, flt_freq, DF)

# ---------------- HEADLINE ----------------
c1, c2, c3 = st.columns(3)
c1.metric("ASW Spread", f"{spread*1e4:.2f} bps")
c2.metric("Dirty price", f"{dirty_price:.4f}")
upfront = details["upfront"]
direction = "Investor paie MM" if upfront > 0 else ("MM paie Investor" if upfront < 0 else "Aucune")
c3.metric("Soulte (upfront)", f"{upfront:+.4f}", help=direction)
st.caption(f"Direction soulte : **{direction}**")

# ---------------- CHART 1 : courbe ----------------
st.subheader("Courbe zéro coupon")
ts = np.linspace(0.01, max(tenors), 100)
zs = [np.interp(t, tenors, rates) * 100 for t in ts]
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ts, y=zs, mode="lines", name="Zero rate"))
fig1.add_trace(go.Scatter(x=tenors, y=[r*100 for r in rates],
                          mode="markers", name="Pillars", marker=dict(size=10)))
fig1.update_layout(xaxis_title="Maturité (années)", yaxis_title="Taux (%)",
                   height=350)
st.plotly_chart(fig1, use_container_width=True)

# ---------------- CHART 2 : cashflows ----------------
st.subheader("Cash flows du swap")
cf_df = cashflow_table(coupon_rate, maturity, fix_freq, flt_freq, spread, DF)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=cf_df["t (années)"], y=cf_df["CF fixe reçu"],
                      name="Fixe reçu (MM→Inv)", marker_color="green"))
fig2.add_trace(go.Bar(x=cf_df["t (années)"], y=cf_df["CF flottant payé"],
                      name="Libor+S payé (Inv→MM)", marker_color="red"))
fig2.update_layout(barmode="relative", xaxis_title="t (années)",
                   yaxis_title="Cash flow", height=400)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- CHART 3 : sensibilité ----------------
st.subheader("Sensibilité ASW Spread vs Dirty Price")
prices = np.linspace(90, 110, 41)
spreads = []
for p in prices:
    s, _ = asw_spread(coupon_rate, maturity, p, fix_freq, flt_freq, DF)
    spreads.append(s * 1e4)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=prices, y=spreads, mode="lines", name="ASW spread"))
fig3.add_vline(x=100, line_dash="dash", line_color="gray",
               annotation_text="Pair (soulte = 0)")
fig3.add_vline(x=dirty_price, line_dash="dot", line_color="blue",
               annotation_text="Prix actuel")
fig3.update_layout(xaxis_title="Dirty price", yaxis_title="ASW spread (bps)",
                   height=400)
st.plotly_chart(fig3, use_container_width=True)

# ---------------- TABLE ----------------
st.subheader("Détail des cash flows")
st.dataframe(cf_df, use_container_width=True)

with st.expander("Décomposition du pricing"):
    st.write({
        "PV jambe fixe": round(details["pv_fix"], 6),
        "PV Libor (sans spread)": round(details["pv_flt_libor"], 6),
        "Annuité flottante (PV01-like)": round(details["annuity_flt"], 6),
        "Soulte (P_dirty - 100)": round(details["upfront"], 6),
        "Spread (bps)": round(spread * 1e4, 4),
    })
