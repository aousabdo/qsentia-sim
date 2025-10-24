# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st
from finance.params import default_params
from finance.sim import run_simulation, summarize_results, run_mc, sustainability_grid
import plotly.express as px


def _freeze_params(params: dict):
    """Stable, hashable representation for caching keys."""
    return tuple(sorted(params.items()))


@st.cache_data(show_spinner=False)
def _compute_sustainability_grid(params_key, aum_vals, hc_vals, n_paths, metric, be_months):
    params = dict(params_key)
    return sustainability_grid(
        params,
        list(aum_vals),
        list(hc_vals),
        n_paths=int(n_paths),
        metric=metric,
        be_months=int(be_months),
    )

st.set_page_config(page_title="QSentia Simulator", layout="wide")
st.title("QSentia — Equity & Fund Economics Simulator")

BASE_DEFAULTS = default_params()

# Tabs
tab_sim, tab_map, tab_about = st.tabs(["💹 Simulator", "🧭 Sustainability Map", "ℹ️ About"])

# ---------------------------------------------------------------------------
# 💹 SIMULATOR TAB
# ---------------------------------------------------------------------------
with tab_sim:
    with st.sidebar:
        st.header("Time & Monte Carlo")
        default_horizon = int(BASE_DEFAULTS["monthly_steps"] / 12)
        horizon_years = st.slider("Horizon (years)", 1, 10, default_horizon, 1)
        monthly_steps = horizon_years * 12
        do_mc = st.checkbox("Enable Monte Carlo (N paths)", value=False)
        mc_n = st.number_input("MC paths", min_value=50, max_value=2000, value=250, step=50)

    st.subheader("Company & Team")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Equity split (always sums to 100%)")
        auto_balance = st.toggle("Lock total to 100% (auto-balance)", value=True)

        founders_equity = st.slider(
            "Founders equity (%)",
            0,
            100,
            int(BASE_DEFAULTS["founders_equity"] * 100),
            1,
        )
        employee_equity = st.slider(
            "Employee pool (%)",
            0,
            100,
            int(BASE_DEFAULTS["employee_equity"] * 100),
            1,
        )

        if auto_balance:
            investor_equity = max(0, 100 - founders_equity - employee_equity)
            st.text_input("Investor equity (%)", value=str(investor_equity), disabled=True)
        else:
            investor_equity = st.slider(
                "Investor equity (%)",
                0,
                100,
                int(BASE_DEFAULTS["investor_equity"] * 100),
                1,
            )

    with col2:
        headcount = st.slider("Initial headcount", 3, 15, int(BASE_DEFAULTS["headcount"]))
        avg_salary = st.number_input(
            "Avg fully-loaded salary per FTE ($/yr)",
            60_000,
            600_000,
            int(BASE_DEFAULTS["avg_salary"]),
            10_000,
        )
        llm_cost = st.number_input(
            "LLM/Data/API ($/mo)",
            0,
            50_000,
            int(BASE_DEFAULTS["llm_cost"]),
            100,
        )
        cloud_cost = st.number_input(
            "Cloud/Infra ($/mo)",
            0,
            50_000,
            int(BASE_DEFAULTS["cloud_cost"]),
            100,
        )
        other_opex = st.number_input(
            "Other Opex ($/mo)",
            0,
            100_000,
            int(BASE_DEFAULTS["other_opex"]),
            100,
        )

    if not auto_balance:
        total = founders_equity + employee_equity + investor_equity
        if abs(total - 100) > 0.0001:
            st.warning(f"Equity currently sums to {total:.0f}% — adjust to 100% or toggle auto-balance.")
            st.stop()

    st.subheader("Fund Structure & Fees")
    col3, col4, col5 = st.columns(3)
    with col3:
        equity_only_investors = st.checkbox(
            "Equity-only investors (no share of carry/fees)",
            value=BASE_DEFAULTS["equity_only_investors"],
        )
        mgmt_fee_pct = st.slider(
            "Management fee (% of AUM/yr)",
            0.0,
            5.0,
            float(BASE_DEFAULTS["mgmt_fee_pct"] * 100),
            0.1,
        )
        carry_pct = st.slider(
            "Performance fee / Carry (%)",
            0.0,
            30.0,
            float(BASE_DEFAULTS["carry_pct"] * 100),
            1.0,
        )
    with col4:
        seed_company_cash = st.number_input(
            "Seed into Company ($)",
            0,
            20_000_000,
            int(BASE_DEFAULTS["seed_company_cash"]),
            50_000,
        )
        seed_fund_aum = st.number_input(
            "Seed Fund AUM ($)",
            0,
            50_000_000,
            int(BASE_DEFAULTS["seed_fund_aum"]),
            100_000,
        )
        mgmt_fee_paid_monthly = st.checkbox(
            "Collect management fee monthly",
            value=BASE_DEFAULTS["mgmt_fee_paid_monthly"],
        )
        carry_high_water_mark = st.checkbox(
            "High-water mark on carry",
            value=BASE_DEFAULTS["carry_high_water_mark"],
        )
    with col5:
        reinvest_carry = st.checkbox(
            "Reinvest carry into company cash",
            value=BASE_DEFAULTS["reinvest_carry"],
        )
        reinvest_mgmt_fee = st.checkbox(
            "Reinvest mgmt fee into company cash",
            value=BASE_DEFAULTS["reinvest_mgmt_fee"],
        )
        hurdle_rate = st.number_input(
            "Hurdle (annual %)",
            0.0,
            10.0,
            float(BASE_DEFAULTS["hurdle_rate"] * 100),
            0.25,
        )

    st.subheader("Performance Assumptions")
    col6, col7 = st.columns(2)
    with col6:
        target_ann_return = st.slider(
            "Target annual return (%)",
            -20.0,
            40.0,
            float(BASE_DEFAULTS["target_ann_return"] * 100),
            0.5,
        )
        target_ann_vol = st.slider(
            "Annual volatility (%)",
            5.0,
            40.0,
            float(BASE_DEFAULTS["target_ann_vol"] * 100),
            0.5,
        )
    with col7:
        rf_rate = st.slider(
            "Risk-free (annual %)",
            0.0,
            5.0,
            float(BASE_DEFAULTS["rf_rate"] * 100),
            0.1,
        )
        growth_new_aum = st.slider(
            "Organic AUM growth (annual %)",
            0.0,
            50.0,
            float(BASE_DEFAULTS["growth_new_aum"] * 100),
            1.0,
        )
        platform_take_on_ext_strats = st.checkbox(
            "Add 2nd strategy Year 2 (+25% AUM)",
            value=BASE_DEFAULTS["platform_take_on_ext_strats"],
        )

    params = dict(BASE_DEFAULTS)
    params.update(dict(
        monthly_steps=monthly_steps,
        founders_equity=founders_equity/100.0,
        employee_equity=employee_equity/100.0,
        investor_equity=investor_equity/100.0,
        headcount=headcount,
        avg_salary=avg_salary,
        llm_cost=llm_cost,
        cloud_cost=cloud_cost,
        other_opex=other_opex,
        seed_company_cash=seed_company_cash,
        seed_fund_aum=seed_fund_aum,
        mgmt_fee_pct=mgmt_fee_pct/100.0,
        carry_pct=carry_pct/100.0,
        equity_only_investors=equity_only_investors,
        mgmt_fee_paid_monthly=mgmt_fee_paid_monthly,
        carry_high_water_mark=carry_high_water_mark,
        reinvest_carry=reinvest_carry,
        reinvest_mgmt_fee=reinvest_mgmt_fee,
        hurdle_rate=hurdle_rate/100.0,
        target_ann_return=target_ann_return/100.0,
        target_ann_vol=target_ann_vol/100.0,
        rf_rate=rf_rate/100.0,
        growth_new_aum=growth_new_aum/100.0,
        platform_take_on_ext_strats=platform_take_on_ext_strats
    ))

    st.divider()

    if do_mc:
        st.info("Monte Carlo enabled: we’ll simulate many paths and show percentiles.")
        mc = run_mc(params, n_paths=int(mc_n), seed=42)
        left, right = st.columns((1, 1))
        with left:
            st.plotly_chart(mc["fund_nav_plot"], use_container_width=True)
        with right:
            st.plotly_chart(mc["company_cash_plot"], use_container_width=True)
        st.plotly_chart(mc["break_even_hist"], use_container_width=True)
        st.dataframe(mc["summary_table"])
    else:
        df = run_simulation(params, seed=42)
        charts, kpis = summarize_results(df, params)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts["fund_nav"], use_container_width=True)
            st.plotly_chart(charts["aum_fee_revenue"], use_container_width=True)
        with c2:
            st.plotly_chart(charts["company_cash"], use_container_width=True)
            st.plotly_chart(charts["runway_burn"], use_container_width=True)

        st.subheader("Key Metrics")
        st.dataframe(kpis)

    st.caption("Tip: Toggle 'Equity-only investors' to enforce your preferred structure (investors get equity, not carry/fees).")

# ---------------------------------------------------------------------------
# 🧭 SUSTAINABILITY MAP TAB
# ---------------------------------------------------------------------------
with tab_map:
    st.header("Sustainability Map (AUM × Headcount)")
    with st.expander("Open heatmap", expanded=True):
        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            aum_min = st.number_input("AUM min ($)", 1_000_000, 1_000_000_000, 5_000_000, 1_000_000)
            aum_max = st.number_input("AUM max ($)", 1_000_000, 1_000_000_000, 15_000_000, 1_000_000)
            n_aum = st.slider("Number of AUM levels", 3, 25, 11)
        with cB:
            hc_min = st.number_input("Headcount min", 2, 100, 4, 1)
            hc_max = st.number_input("Headcount max", 2, 100, 12, 1)
            n_hc = st.slider("Number of headcount levels", 3, 25, 9)
        with cC:
            metric = st.selectbox("Metric", ["No cash-out over horizon", "Break-even ≤ T months"])
            be_months = st.number_input("T (months for break-even)", 3, 60, 24, 1)
            n_paths_grid = st.number_input("MC paths per cell", 50, 2000, 100, 25)

        if aum_max <= aum_min:
            st.error("AUM max must be greater than AUM min.")
            st.stop()
        if hc_max < hc_min:
            st.error("Headcount max must be ≥ headcount min.")
            st.stop()

        def spaced_int_levels(vmin, vmax, n):
            raw = np.linspace(float(vmin), float(vmax), int(n))
            vals = np.unique(np.round(raw).astype(int))
            return vals.tolist()

        aum_vals = spaced_int_levels(aum_min, aum_max, n_aum)
        hc_vals = spaced_int_levels(hc_min, hc_max, n_hc)
        metric_key = "no_cashout" if metric == "No cash-out over horizon" else "break_even"

        grid_inputs = dict(
            params_key=_freeze_params(params),
            aum=tuple(aum_vals),
            hc=tuple(hc_vals),
            n_paths=int(n_paths_grid),
            metric=metric_key,
            be_months=int(be_months)
        )

        stored_grid = st.session_state.get("sustainability_grid")
        run_map = st.button("Run heatmap", key="run_heatmap")

        if run_map:
            with st.spinner("Running Monte Carlo grid…"):
                Z = _compute_sustainability_grid(
                    grid_inputs["params_key"],
                    grid_inputs["aum"],
                    grid_inputs["hc"],
                    grid_inputs["n_paths"],
                    grid_inputs["metric"],
                    grid_inputs["be_months"],
                )
            st.session_state["sustainability_grid"] = {
                "inputs": grid_inputs,
                "matrix": Z,
            }
            stored_grid = st.session_state["sustainability_grid"]
        elif stored_grid and stored_grid["inputs"] != grid_inputs:
            st.info("Inputs changed — click 'Run heatmap' to recompute with the new settings.")

        if stored_grid and stored_grid["inputs"] == grid_inputs:
            Z = stored_grid["matrix"]
            H, W = Z.shape
            aum_vals_plot = aum_vals[:W]
            hc_vals_plot = hc_vals[:H]

            z_pct = 100.0 * Z
            fig = px.imshow(
                z_pct,
                x=[f"${v/1_000_000:.0f}M" for v in aum_vals_plot],
                y=[str(int(h)) for h in hc_vals_plot],
                color_continuous_scale="Viridis",
                origin="lower",
                aspect="auto",
                labels=dict(x="Seed Fund AUM", y="Headcount", color="Prob (%)"),
                title=f"Sustainability: {metric} — Monte Carlo Probability"
            )
            fig.update_traces(hovertemplate="AUM=%{x}<br>HC=%{y}<br>P=%{z:.1f}%")
            fig.update_layout(
                template="qsentia_investor",
                coloraxis_colorbar=dict(title="Prob (%)", ticksuffix="%"),
            )
            st.plotly_chart(fig, use_container_width=True)

            i, j = divmod(int(np.nanargmax(Z)), W)
            best_hc = hc_vals_plot[i]
            best_aum = aum_vals_plot[j]
            best_p = Z[i, j]
            st.success(f"Best region: HC={best_hc}, AUM=${best_aum:,.0f} → P={best_p:.1%}")
            st.caption(
                f"Grid used → AUM levels: {len(aum_vals_plot)} from ${aum_vals_plot[0]:,} to ${aum_vals_plot[-1]:,}; "
                f"HC levels: {len(hc_vals_plot)} from {hc_vals_plot[0]} to {hc_vals_plot[-1]}."
            )
        else:
            st.caption("Adjust inputs and press 'Run heatmap' to generate the sustainability map.")

# ---------------------------------------------------------------------------
# ℹ️ ABOUT TAB
# ---------------------------------------------------------------------------
with tab_about:
    st.header("About the QSentia Simulator")
    st.markdown("""
    ### Purpose
    The QSentia Simulator models the **financial sustainability** of an early-stage AI-driven hedge fund startup.
    It links fund-level performance (returns, volatility, AUM) with management-company operations (team size, salaries, fees).

    ### How It Works
    **Fund simulation:**  
    • Fund begins with a set **AUM (Assets Under Management)**.  
    • Each month returns are drawn from a normal distribution using your target annual return & volatility.  
    • Management and performance fees are applied.  

    **Company simulation:**  
    • The management company starts with seed cash and incurs payroll and infra costs.  
    • Monthly inflows = management fees + carry (if reinvested).  
    • Cumulative cash tracks runway and break-even.  

    **Monte Carlo analysis:**  
    • Optional multiple-path simulation shows variability (P10/P50/P90).  

    ### Sustainability Map
    • Evaluates solvency probability over different **AUM** and **Headcount** combinations.  
    • Each grid cell runs many simulations to estimate P(no cash-out) or P(break-even ≤ T months).  
    • Hotter colors = higher survival probability.  

    ### Dict
    | Term | Meaning |
    |:--|:--|
    | **AUM** | Assets Under Management — total investor capital traded by the fund |
    | **NAV** | Net Asset Value — fund’s current market value |
    | **Mgmt Fee** | % of AUM paid yearly to QSentia for operations |
    | **Carry** | % of fund profits paid to QSentia |
    | **Headcount** | Team size (FTEs) |
    | **Runway** | Time until company cash = 0 |
    | **Break-even** | First month inflows ≥ expenses |

    ### Assumptions
    • Monthly compounding, continuous reinvestment (if toggled).   
    """)
