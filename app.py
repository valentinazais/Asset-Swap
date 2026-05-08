# Asset Swap Pricer — run with: streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import brentq


# =============================================================================
# CURVE
# =============================================================================

def build_discount_factor(tenors: list[float], zero_rates: list[float]):
    """Continuous-compounding zero curve, linear interpolation on rates."""
    ts = np.asarray(tenors, dtype=float)
    rs = np.asarray(zero_rates, dtype=float)

    def DF(t: float) -> float:
        if t <= 0.0:
            return 1.0
        r = float(np.interp(t, ts, rs))
        return float(np.exp(-r * t))

    return DF


def forward_simple(DF, t1: float, t2: float) -> float:
    """Simple-compounded forward between t1 and t2 (tau = t2 - t1)."""
    tau = t2 - t1
    return (DF(t1) / DF(t2) - 1.0) / tau


def curve_dataframe(tenors: list[float], zero_rates: list[float],
                    n_points: int = 100) -> pd.DataFrame:
    ts = np.linspace(min(tenors), max(tenors), n_points)
    rs = np.interp(ts, tenors, zero_rates)
    return pd.DataFrame({"tenor": ts, "zero_rate": rs})


# =============================================================================
# BOND
# =============================================================================

def coupon_schedule(maturity: float, frequency: int) -> list[float]:
    n = int(round(maturity * frequency))
    dt = 1.0 / frequency
    return [(i + 1) * dt for i in range(n)]


def dirty_price_from_ytm(coupon_rate: float, maturity: float, ytm: float,
                          frequency: int, notional: float = 100.0) -> float:
    dates = coupon_schedule(maturity, frequency)
    c = coupon_rate * notional / frequency
    pv = 0.0
    for t in dates:
        pv += c / (1.0 + ytm / frequency) ** (t * frequency)
    pv += notional / (1.0 + ytm / frequency) ** (maturity * frequency)
    return pv


def ytm_from_dirty_price(dirty: float, coupon_rate: float, maturity: float,
                         frequency: int, notional: float = 100.0) -> float:
    def diff(y: float) -> float:
        return dirty_price_from_ytm(coupon_rate, maturity, y, frequency, notional) - dirty
    return brentq(diff, -0.5, 1.0, xtol=1e-10)


# =============================================================================
# PRICING
# =============================================================================

def _fixed_leg_pv(coupon_rate: float, maturity: float, frequency: int,
                  DF, notional: float) -> float:
    dates = coupon_schedule(maturity, frequency)
    tau = 1.0 / frequency
    return sum(coupon_rate * notional * tau * DF(t) for t in dates)


def _floating_leg_components(maturity: float, frequency: int, DF,
                              notional: float) -> tuple[float, float]:
    """Return (PV of Libor leg without spread, floating annuity)."""
    dates = coupon_schedule(maturity, frequency)
    tau = 1.0 / frequency
    pv_libor = 0.0
    annuity = 0.0
    t_prev = 0.0
    for t in dates:
        L = forward_simple(DF, t_prev, t)
        pv_libor += L * notional * tau * DF(t)
        annuity += tau * DF(t)
        t_prev = t
    return pv_libor, annuity


def par_asw_spread(coupon_rate: float, maturity: float, dirty_price: float,
                   fix_frequency: int, flt_frequency: int, DF,
                   notional: float = 100.0) -> float:
    pv_fix = _fixed_leg_pv(coupon_rate, maturity, fix_frequency, DF, notional)
    pv_libor, annuity_flt = _floating_leg_components(
        maturity, flt_frequency, DF, notional
    )
    upfront = dirty_price - notional
    return (pv_fix - pv_libor - upfront) / (annuity_flt * notional)


def pricing_report(coupon_rate: float, maturity: float, dirty_price: float,
                   fix_frequency: int, flt_frequency: int, DF,
                   notional: float = 100.0,
                   market_spread_bps: float | None = None) -> dict:
    pv_fix = _fixed_leg_pv(coupon_rate, maturity, fix_frequency, DF, notional)
    pv_libor, annuity_flt = _floating_leg_components(
        maturity, flt_frequency, DF, notional
    )
    upfront = dirty_price - notional
    spread = (pv_fix - pv_libor - upfront) / (annuity_flt * notional)

    out = {
        "fair_spread_bps": spread * 1e4,
        "upfront": upfront,
        "premium_leg_pv": pv_fix,
        "libor_leg_pv": pv_libor,
        "floating_annuity": annuity_flt,
        "pv01_floating": annuity_flt * notional * 1e-4,
        "dirty_price": dirty_price,
    }

    if market_spread_bps is not None:
        market_spread = market_spread_bps / 1e4
        mtm = (spread - market_spread) * annuity_flt * notional
        out["market_spread_bps"] = market_spread_bps
        out["mtm"] = mtm

    return out


# =============================================================================
# ANALYTICS
# =============================================================================

def cashflow_table(coupon_rate: float, maturity: float, fix_frequency: int,
                    flt_frequency: int, spread: float, DF,
                    notional: float = 100.0) -> pd.DataFrame:
    fix_dates = set(np.round(coupon_schedule(maturity, fix_frequency), 8))
    flt_dates = coupon_schedule(maturity, flt_frequency)
    tau_fix = 1.0 / fix_frequency
    tau_flt = 1.0 / flt_frequency

    rows = []
    t_prev = 0.0
    for t in flt_dates:
        L = forward_simple(DF, t_prev, t)
        cf_flt = -(L + spread) * notional * tau_flt
        cf_fix = (coupon_rate * notional * tau_fix
                  if round(t, 8) in fix_dates else 0.0)
        rows.append({
            "t": t,
            "DF": DF(t),
            "libor_fwd": L,
            "cf_fix_received": cf_fix,
            "cf_flt_paid": cf_flt,
            "cf_net": cf_fix + cf_flt,
        })
        t_prev = t

    return pd.DataFrame(rows)


def price_sensitivity(coupon_rate: float, maturity: float,
                       fix_frequency: int, flt_frequency: int, DF,
                       price_min: float = 90.0, price_max: float = 110.0,
                       n: int = 41, notional: float = 100.0) -> pd.DataFrame:
    prices = np.linspace(price_min, price_max, n)
    spreads = [par_asw_spread(coupon_rate, maturity, p, fix_frequency,
                               flt_frequency, DF, notional) * 1e4
               for p in prices]
    return pd.DataFrame({"dirty_price": prices, "spread_bps": spreads})


def maturity_sensitivity(coupon_rate: float, dirty_price: float,
                          fix_frequency: int, flt_frequency: int, DF,
                          mat_min: float = 1.0, mat_max: float = 10.0,
                          n: int = 19, notional: float = 100.0) -> pd.DataFrame:
    mats = np.linspace(mat_min, mat_max, n)
    spreads = [par_asw_spread(coupon_rate, m, dirty_price, fix_frequency,
                               flt_frequency, DF, notional) * 1e4
               for m in mats]
    return pd.DataFrame({"maturity": mats, "spread_bps": spreads})


# =============================================================================
# PLOTS
# =============================================================================

_LAYOUT = dict(
    template="plotly_white",
    height=400,
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", y=-0.2),
)


def plot_zero_curve(curve_df: pd.DataFrame,
                     pillars: list[tuple[float, float]] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["tenor"], y=curve_df["zero_rate"] * 100,
        mode="lines", name="Zero rate", line=dict(width=2.5),
    ))
    if pillars:
        xs, ys = zip(*pillars)
        fig.add_trace(go.Scatter(
            x=list(xs), y=[y * 100 for y in ys],
            mode="markers", name="Pillars",
            marker=dict(size=10, symbol="diamond"),
        ))
    fig.update_layout(title="Zero-Coupon Curve",
                      xaxis_title="Maturity (years)",
                      yaxis_title="Rate (%)", **_LAYOUT)
    return fig


def plot_cashflows(cf_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cf_df["t"], y=cf_df["cf_fix_received"],
        name="Fixed received (MM → Investor)", marker_color="#2ca02c",
    ))
    fig.add_trace(go.Bar(
        x=cf_df["t"], y=cf_df["cf_flt_paid"],
        name="Libor + S paid (Investor → MM)", marker_color="#d62728",
    ))
    fig.add_trace(go.Scatter(
        x=cf_df["t"], y=cf_df["cf_net"],
        name="Net cashflow", mode="lines+markers",
        line=dict(color="black", dash="dot"),
    ))
    fig.update_layout(title="Swap Leg Cashflows (Investor View)",
                      xaxis_title="Time (years)",
                      yaxis_title="Cashflow",
                      barmode="relative", **_LAYOUT)
    return fig


def plot_price_sensitivity(sens_df: pd.DataFrame, current_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens_df["dirty_price"], y=sens_df["spread_bps"],
        mode="lines", name="ASW spread", line=dict(width=2.5),
    ))
    fig.add_vline(x=100, line_dash="dash", line_color="gray",
                  annotation_text="Par (no upfront)")
    fig.add_vline(x=current_price, line_dash="dot", line_color="blue",
                  annotation_text="Current price")
    fig.update_layout(title="ASW Spread vs Dirty Price",
                      xaxis_title="Dirty price",
                      yaxis_title="ASW spread (bps)", **_LAYOUT)
    return fig


def plot_maturity_sensitivity(sens_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens_df["maturity"], y=sens_df["spread_bps"],
        mode="lines+markers", name="ASW spread", line=dict(width=2.5),
    ))
    fig.update_layout(title="ASW Spread vs Bond Maturity",
                      xaxis_title="Maturity (years)",
                      yaxis_title="ASW spread (bps)", **_LAYOUT)
    return fig


def plot_leg_decomposition(pv_fix: float, pv_libor: float,
                            upfront: float, spread_pv: float) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["PV Fixed Leg", "-PV Libor Leg", "-Upfront", "PV Spread Leg"],
        y=[pv_fix, -pv_libor, -upfront, spread_pv],
        connector=dict(line=dict(color="gray")),
    ))
    fig.update_layout(title="Asset Swap Package Decomposition",
                      yaxis_title="PV", **_LAYOUT)
    return fig


# =============================================================================
# APP
# =============================================================================

st.set_page_config(
    page_title="Fixed Income Tool",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Asset Swap Pricer")

tab1, tab2, tab3 = st.tabs(["Pricer", "Sensitivities", "Curve & Cashflows"])


# -- Tab 1: Pricer --

with tab1:
    st.markdown("### Par Asset Swap Pricing")
    st.markdown(
        "Price a par asset swap package: a fixed-rate bond combined with an "
        "interest rate swap converting fixed coupons into Libor + spread. "
        "The **ASW spread** compensates the investor for the issuer's credit risk."
    )

    st.markdown("---")
    st.markdown("#### Bond & Market")

    if "pr_reset_counter" not in st.session_state:
        st.session_state.pr_reset_counter = 0
    pr_rc = st.session_state.pr_reset_counter

    if st.button("↻ Reset to Defaults", key="pr_reset"):
        st.session_state.pr_reset_counter = pr_rc + 1
        st.rerun()

    p_c1, p_c2, p_c3 = st.columns(3)
    notional = p_c1.number_input(
        "Notional", value=10_000_000, step=1_000_000,
        format="%d", help="Bond face value", key=f"pr_notional_{pr_rc}",
    )
    maturity = p_c2.number_input(
        "Maturity (years)", value=5.0, step=0.5,
        min_value=0.5, max_value=30.0, format="%.1f", key=f"pr_maturity_{pr_rc}",
    )
    coupon_rate = p_c3.number_input(
        "Bond Coupon (%)", value=4.0, step=0.10,
        min_value=0.0, max_value=20.0, format="%.2f",
        key=f"pr_coupon_{pr_rc}",
    ) / 100.0

    p_c4, p_c5, p_c6 = st.columns(3)
    fix_frequency = p_c4.selectbox(
        "Bond Coupon Frequency", [1, 2, 4],
        index=0, format_func=lambda x: {
            1: "Annual", 2: "Semi-annual", 4: "Quarterly"
        }[x], key=f"pr_fixfreq_{pr_rc}",
    )
    flt_frequency = p_c5.selectbox(
        "Floating Leg Frequency", [2, 4, 12],
        index=1, format_func=lambda x: {
            2: "Semi-annual", 4: "Quarterly", 12: "Monthly"
        }[x], key=f"pr_fltfreq_{pr_rc}",
    )
    price_mode = p_c6.radio(
        "Price input mode", ["YTM", "Dirty Price"],
        horizontal=True, key=f"pr_pmode_{pr_rc}",
    )

    p_c7, p_c8 = st.columns(2)
    if price_mode == "YTM":
        ytm = p_c7.slider(
            "Yield to Maturity (%)", min_value=-1.0, max_value=15.0,
            value=3.5, step=0.05, format="%.2f", key=f"pr_ytm_{pr_rc}",
        ) / 100.0
        dirty_price = dirty_price_from_ytm(coupon_rate, maturity, ytm,
                                            fix_frequency, notional=100.0)
        p_c8.info(f"Implied Dirty Price = {dirty_price:.4f}")
    else:
        dirty_price = p_c7.number_input(
            "Dirty Price (% of par)", value=102.00, step=0.10,
            min_value=50.0, max_value=200.0, format="%.4f",
            key=f"pr_dp_{pr_rc}",
        )
        try:
            ytm = ytm_from_dirty_price(dirty_price, coupon_rate, maturity,
                                       fix_frequency, notional=100.0)
            p_c8.info(f"Implied YTM = {ytm * 100:.4f}%")
        except Exception:
            ytm = float("nan")
            p_c8.warning("Could not invert dirty price to YTM")

    st.markdown("#### Discount Curve")

    default_curve = pd.DataFrame({
        "tenor": [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "rate (%)": [2.20, 2.50, 2.80, 3.00, 3.20, 3.40, 3.50],
    })
    curve_df = st.data_editor(
        default_curve, num_rows="dynamic", width="stretch",
        key=f"pr_curve_{pr_rc}",
    )
    tenors = curve_df["tenor"].tolist()
    zero_rates = (curve_df["rate (%)"] / 100.0).tolist()
    DF = build_discount_factor(tenors, zero_rates)

    st.markdown("#### Mark to Market")
    mtm_c1, mtm_c2 = st.columns(2)
    enable_mtm = mtm_c1.checkbox("Enable MTM calculation", value=False,
                                  key=f"pr_mtm_en_{pr_rc}")
    market_spread_bps = None
    if enable_mtm:
        contractual_spread = par_asw_spread(coupon_rate, maturity, dirty_price,
                                             fix_frequency, flt_frequency,
                                             DF, notional=100.0) * 1e4
        market_spread_bps = mtm_c2.number_input(
            "Current Market ASW Spread (bps)",
            value=float(round(contractual_spread + 20.0, 1)),
            step=1.0, key=f"pr_mkt_{pr_rc}",
        )

    st.markdown("---")
    st.markdown("#### Pricing Results")

    report = pricing_report(
        coupon_rate=coupon_rate,
        maturity=maturity,
        dirty_price=dirty_price,
        fix_frequency=fix_frequency,
        flt_frequency=flt_frequency,
        DF=DF,
        notional=100.0,
        market_spread_bps=market_spread_bps,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("ASW Spread", f"{report['fair_spread_bps']:.2f} bps")
    col2.metric("Dirty Price", f"{report['dirty_price']:.4f}")
    upfront = report["upfront"]
    direction = ("Investor → MM" if upfront > 0
                 else "MM → Investor" if upfront < 0 else "None")
    col3.metric("Upfront (par)", f"{upfront:+.4f}", help=f"Direction: {direction}")

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("PV Fixed Leg", f"{report['premium_leg_pv']:.4f}")
    col5.metric("PV Libor Leg", f"{report['libor_leg_pv']:.4f}")
    col6.metric("Floating Annuity", f"{report['floating_annuity']:.4f}")
    col7.metric("PV01 Floating", f"{report['pv01_floating']:.6f}")

    if enable_mtm and "mtm" in report:
        st.markdown("#### MTM (existing ASW position)")
        mtm_per100 = report["mtm"]
        mtm_total = mtm_per100 / 100.0 * notional
        m1, m2 = st.columns(2)
        m1.metric("MTM (% of par)", f"{mtm_per100:+.4f}")
        m2.metric("MTM (notional units)", f"{mtm_total:+,.2f}")

    st.markdown("---")
    st.markdown("#### Package Decomposition")
    spread_pv = (report["fair_spread_bps"] / 1e4) * report["floating_annuity"] * 100.0
    st.plotly_chart(
        plot_leg_decomposition(
            pv_fix=report["premium_leg_pv"],
            pv_libor=report["libor_leg_pv"],
            upfront=report["upfront"],
            spread_pv=spread_pv,
        ),
        width="stretch",
    )


# -- Tab 2: Sensitivities --

with tab2:
    st.markdown("### Sensitivity Analysis")
    st.markdown(
        "Explore how the ASW spread reacts to bond price and maturity, "
        "holding the discount curve fixed."
    )

    st.markdown("---")
    st.markdown("#### Sensitivity Setup")

    if "se_reset_counter" not in st.session_state:
        st.session_state.se_reset_counter = 0
    se_rc = st.session_state.se_reset_counter

    if st.button("↻ Reset to Defaults", key="se_reset"):
        st.session_state.se_reset_counter = se_rc + 1
        st.rerun()

    se_c1, se_c2, se_c3 = st.columns(3)
    se_coupon = se_c1.number_input(
        "Coupon (%)", value=4.0, step=0.1, format="%.2f",
        key=f"se_coupon_{se_rc}",
    ) / 100.0
    se_maturity = se_c2.number_input(
        "Maturity (years)", value=5.0, step=0.5,
        min_value=0.5, max_value=30.0, format="%.1f",
        key=f"se_mat_{se_rc}",
    )
    se_dirty = se_c3.number_input(
        "Reference Dirty Price", value=102.0, step=0.5,
        min_value=50.0, max_value=200.0, format="%.2f",
        key=f"se_dp_{se_rc}",
    )

    se_c4, se_c5 = st.columns(2)
    se_fix_freq = se_c4.selectbox(
        "Bond Frequency", [1, 2, 4], index=0,
        format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}[x],
        key=f"se_fix_{se_rc}",
    )
    se_flt_freq = se_c5.selectbox(
        "Floating Frequency", [2, 4, 12], index=1,
        format_func=lambda x: {2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}[x],
        key=f"se_flt_{se_rc}",
    )

    st.markdown("#### Discount Curve")
    se_default_curve = pd.DataFrame({
        "tenor": [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "rate (%)": [2.20, 2.50, 2.80, 3.00, 3.20, 3.40, 3.50],
    })
    se_curve_df = st.data_editor(
        se_default_curve, num_rows="dynamic", width="stretch",
        key=f"se_curve_{se_rc}",
    )
    se_tenors = se_curve_df["tenor"].tolist()
    se_rates = (se_curve_df["rate (%)"] / 100.0).tolist()
    se_DF = build_discount_factor(se_tenors, se_rates)

    st.markdown("---")
    st.markdown("#### Results")

    try:
        price_df = price_sensitivity(
            coupon_rate=se_coupon, maturity=se_maturity,
            fix_frequency=se_fix_freq, flt_frequency=se_flt_freq,
            DF=se_DF,
        )
        mat_df = maturity_sensitivity(
            coupon_rate=se_coupon, dirty_price=se_dirty,
            fix_frequency=se_fix_freq, flt_frequency=se_flt_freq,
            DF=se_DF,
            mat_max=min(max(se_tenors), 15.0),
        )

        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(
                plot_price_sensitivity(price_df, current_price=se_dirty),
                width="stretch",
            )
        with ch2:
            st.plotly_chart(
                plot_maturity_sensitivity(mat_df),
                width="stretch",
            )

        st.markdown("#### Spread vs Price Table")
        display_price = pd.DataFrame({
            "Dirty Price": [f"{p:.2f}" for p in price_df["dirty_price"]],
            "ASW Spread (bps)": [f"{s:.2f}" for s in price_df["spread_bps"]],
        })
        st.dataframe(display_price, width="stretch", hide_index=True)

    except Exception as exc:
        st.error(f"Sensitivity computation failed: {exc}")


# -- Tab 3: Curve & Cashflows --

with tab3:
    st.markdown("### Curve & Cashflow Inspection")
    st.markdown(
        "Visualise the zero-coupon discount curve and the per-period swap "
        "cashflows used in the ASW spread calculation."
    )

    st.markdown("---")
    st.markdown("#### Setup")

    if "cf_reset_counter" not in st.session_state:
        st.session_state.cf_reset_counter = 0
    cf_rc = st.session_state.cf_reset_counter

    if st.button("↻ Reset to Defaults", key="cf_reset"):
        st.session_state.cf_reset_counter = cf_rc + 1
        st.rerun()

    cf_c1, cf_c2, cf_c3 = st.columns(3)
    cf_coupon = cf_c1.number_input(
        "Coupon (%)", value=4.0, step=0.1, format="%.2f",
        key=f"cf_coupon_{cf_rc}",
    ) / 100.0
    cf_maturity = cf_c2.number_input(
        "Maturity (years)", value=5.0, step=0.5,
        min_value=0.5, max_value=30.0, format="%.1f",
        key=f"cf_mat_{cf_rc}",
    )
    cf_dirty = cf_c3.number_input(
        "Dirty Price", value=102.0, step=0.5,
        min_value=50.0, max_value=200.0, format="%.2f",
        key=f"cf_dp_{cf_rc}",
    )

    cf_c4, cf_c5 = st.columns(2)
    cf_fix_freq = cf_c4.selectbox(
        "Bond Frequency", [1, 2, 4], index=0,
        format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}[x],
        key=f"cf_fix_{cf_rc}",
    )
    cf_flt_freq = cf_c5.selectbox(
        "Floating Frequency", [2, 4, 12], index=1,
        format_func=lambda x: {2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}[x],
        key=f"cf_flt_{cf_rc}",
    )

    st.markdown("#### Discount Curve")
    cf_default_curve = pd.DataFrame({
        "tenor": [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "rate (%)": [2.20, 2.50, 2.80, 3.00, 3.20, 3.40, 3.50],
    })
    cf_curve_df = st.data_editor(
        cf_default_curve, num_rows="dynamic", width="stretch",
        key=f"cf_curve_{cf_rc}",
    )
    cf_tenors = cf_curve_df["tenor"].tolist()
    cf_rates = (cf_curve_df["rate (%)"] / 100.0).tolist()
    cf_DF = build_discount_factor(cf_tenors, cf_rates)

    st.markdown("---")

    try:
        spread = par_asw_spread(
            cf_coupon, cf_maturity, cf_dirty,
            cf_fix_freq, cf_flt_freq, cf_DF,
        )

        ch1, ch2 = st.columns(2)
        with ch1:
            dense_curve = curve_dataframe(cf_tenors, cf_rates)
            pillars = list(zip(cf_tenors, cf_rates))
            st.plotly_chart(
                plot_zero_curve(dense_curve, pillars=pillars),
                width="stretch",
            )
        with ch2:
            cf_df = cashflow_table(
                cf_coupon, cf_maturity, cf_fix_freq, cf_flt_freq,
                spread, cf_DF,
            )
            st.plotly_chart(plot_cashflows(cf_df), width="stretch")

        st.markdown("#### Cashflow Detail")
        cf_display = pd.DataFrame({
            "t (years)": [f"{t:.4f}" for t in cf_df["t"]],
            "DF": [f"{d:.6f}" for d in cf_df["DF"]],
            "Libor fwd (%)": [f"{l*100:.4f}" for l in cf_df["libor_fwd"]],
            "CF Fixed Received": [f"{x:.4f}" for x in cf_df["cf_fix_received"]],
            "CF Floating Paid": [f"{x:.4f}" for x in cf_df["cf_flt_paid"]],
            "Net CF": [f"{x:.4f}" for x in cf_df["cf_net"]],
        })
        st.dataframe(cf_display, width="stretch", hide_index=True)

        st.caption(f"Spread used in floating leg: **{spread*1e4:.2f} bps**")

    except Exception as exc:
        st.error(f"Cashflow computation failed: {exc}")
