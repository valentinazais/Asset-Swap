from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


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
        mode="lines", name="Zero rate",
        line=dict(width=2.5),
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
                      yaxis_title="Rate (%)",
                      **_LAYOUT)
    return fig


def plot_cashflows(cf_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cf_df["t"], y=cf_df["cf_fix_received"],
        name="Fixed received (MM → Investor)",
        marker_color="#2ca02c",
    ))
    fig.add_trace(go.Bar(
        x=cf_df["t"], y=cf_df["cf_flt_paid"],
        name="Libor + S paid (Investor → MM)",
        marker_color="#d62728",
    ))
    fig.add_trace(go.Scatter(
        x=cf_df["t"], y=cf_df["cf_net"],
        name="Net cashflow", mode="lines+markers",
        line=dict(color="black", dash="dot"),
    ))
    fig.update_layout(title="Swap Leg Cashflows (Investor View)",
                      xaxis_title="Time (years)",
                      yaxis_title="Cashflow",
                      barmode="relative",
                      **_LAYOUT)
    return fig


def plot_price_sensitivity(sens_df: pd.DataFrame,
                            current_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens_df["dirty_price"], y=sens_df["spread_bps"],
        mode="lines", name="ASW spread",
        line=dict(width=2.5),
    ))
    fig.add_vline(x=100, line_dash="dash", line_color="gray",
                  annotation_text="Par (no upfront)")
    fig.add_vline(x=current_price, line_dash="dot", line_color="blue",
                  annotation_text="Current price")
    fig.update_layout(title="ASW Spread vs Dirty Price",
                      xaxis_title="Dirty price",
                      yaxis_title="ASW spread (bps)",
                      **_LAYOUT)
    return fig


def plot_maturity_sensitivity(sens_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens_df["maturity"], y=sens_df["spread_bps"],
        mode="lines+markers", name="ASW spread",
        line=dict(width=2.5),
    ))
    fig.update_layout(title="ASW Spread vs Bond Maturity",
                      xaxis_title="Maturity (years)",
                      yaxis_title="ASW spread (bps)",
                      **_LAYOUT)
    return fig


def plot_leg_decomposition(pv_fix: float, pv_libor: float,
                            upfront: float, spread_pv: float) -> go.Figure:
    """Waterfall showing how the spread closes the package value."""
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["PV Fixed Leg", "-PV Libor Leg", "-Upfront", "PV Spread Leg"],
        y=[pv_fix, -pv_libor, -upfront, spread_pv],
        connector=dict(line=dict(color="gray")),
    ))
    fig.update_layout(title="Asset Swap Package Decomposition",
                      yaxis_title="PV",
                      **_LAYOUT)
    return fig
