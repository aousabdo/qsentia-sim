# finance/sim.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def monthly_return_series(ann_mu, ann_vol, steps, seed=None):
    rng = np.random.default_rng(seed)
    mu_m = (1 + ann_mu) ** (1/12) - 1
    vol_m = ann_vol / np.sqrt(12)
    rets = rng.normal(mu_m, vol_m, steps)
    return rets

def apply_hwm_and_carry(nav, carry_pct, hurdle_rate=0.0):
    """Monthly HWM carry with optional hurdle. nav is series of fund NAV."""
    hwm = nav[0]
    carry = np.zeros_like(nav)
    for t in range(1, len(nav)):
        hurdle_level = hwm * (1 + (hurdle_rate/12.0))
        if nav[t] > hurdle_level:
            perf = nav[t] - max(hwm, hurdle_level)
            carry[t] = carry_pct * perf
            hwm = nav[t]  # new high-water mark after carry
        else:
            hwm = max(hwm, nav[t])
    return carry

def run_simulation(p, seed=None):
    steps = int(p["monthly_steps"])
    # idx = pd.period_range(freq="M", periods=steps).to_timestamp()
    # NEW: build a clean month-end index
    start = p.get("start_date", None)
    if start is None:
        # start at the end of the current month
        start = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
    else:
        start = pd.to_datetime(start)

    idx = pd.date_range(start=start, periods=steps, freq="M")  # month-end stamps

    # Company state
    company_cash = np.zeros(steps, dtype=float)
    company_cash[0] = p["seed_company_cash"]

    # Fund state
    aum = np.zeros(steps, dtype=float)
    aum[0] = p["seed_fund_aum"]
    nav = np.zeros(steps, dtype=float)
    nav[0] = aum[0]  # NAV starts equal to AUM (simple model)

    # Operating costs
    monthly_payroll = p["headcount"] * (p["avg_salary"] / 12.0)
    fixed_opex = p["llm_cost"] + p["cloud_cost"] + p["other_opex"]
    monthly_burn = monthly_payroll + fixed_opex

    # Fees
    mgmt_fee_m = p["mgmt_fee_pct"] / 12.0  # % of AUM per month
    carry_pct = p["carry_pct"]

    # Returns path
    rets = monthly_return_series(p["target_ann_return"], p["target_ann_vol"], steps, seed=seed)

    # Organic AUM growth (subscriptions) + optional platform uplift Year 2
    subs_m = (1 + p["growth_new_aum"]) ** (1/12) - 1
    uplift_month = 12
    uplift_factor = 1.25 if p["platform_take_on_ext_strats"] else 1.0

    mgmt_fees = np.zeros(steps, dtype=float)
    carry_paid = np.zeros(steps, dtype=float)

    for t in range(1, steps):
        # AUM growth from subscriptions
        aum[t] = aum[t-1] * (1 + subs_m)
        if t == uplift_month:
            aum[t] *= uplift_factor

        # NAV evolves with returns on prior NAV
        nav[t] = (nav[t-1]) * (1 + rets[t])

        # Management fee (paid monthly if set; otherwise accrue)
        mgmt_fees[t] = mgmt_fee_m * aum[t]
        if p["mgmt_fee_paid_monthly"]:
            nav[t] -= mgmt_fees[t]  # fees flow out of fund

        # Carry (with optional HWM/hurdle, applied on-the-fly monthly)
        # For tractability, compute carry after fees effect on NAV:
        # We'll compute an ex-post carry series against the current nav path
    # Compute carry with HWM/hurdle on the full nav
    carry_series = apply_hwm_and_carry(nav.copy(), carry_pct, p["hurdle_rate"]) if p["carry_high_water_mark"] else np.maximum(nav - nav.cummax(), 0) * carry_pct
    carry_paid = carry_series

    # Deduct carry from NAV where applicable
    nav = nav - carry_paid

    # Company cash flows (fees -> company; investors get equity only if equity_only_investors=True)
    for t in range(1, steps):
        inflow = 0.0
        if p["reinvest_mgmt_fee"]:
            inflow += mgmt_fees[t]
        if p["reinvest_carry"]:
            inflow += carry_paid[t]
        company_cash[t] = company_cash[t-1] + inflow - (monthly_burn)

    df = pd.DataFrame({
        "date": idx,
        "fund_nav": nav,
        "fund_aum": aum,
        "mgmt_fees": mgmt_fees,
        "carry_paid": carry_paid,
        "company_cash": company_cash,
        "monthly_burn": monthly_burn,
        "monthly_return": rets
    }).set_index("date")

    # Simple break-even = first month where inflow >= burn on a trailing basis
    df["company_inflow"] = (p["reinvest_mgmt_fee"] * df["mgmt_fees"]) + (p["reinvest_carry"] * df["carry_paid"])
    df["net_cashflow"] = df["company_inflow"] - df["monthly_burn"]
    df["cum_cash"] = p["seed_company_cash"] + df["net_cashflow"].cumsum()

    return df

def summarize_results(df, p):
    # KPIs
    runway_months = int(np.argmax(df["cum_cash"].values <= 0))
    if runway_months == 0 and df["cum_cash"].iloc[0] > 0:
        runway_text = "No depletion in horizon"
    elif runway_months == 0:
        runway_text = "Depleted immediately"
    else:
        runway_text = f"{runway_months} months (until cash ≤ 0)"

    # Break-even (first month inflow >= burn)
    be_idx = np.where(df["company_inflow"].values >= df["monthly_burn"].values)[0]
    break_even = "Not reached"
    if len(be_idx) > 0:
        break_even = df.index[be_idx[0]].strftime("%Y-%m")

    # Estimated “management co.” value (very rough): 4× annualized fees + 20% of annualized carry
    ann_mgmt = df["mgmt_fees"].mean() * 12
    ann_carry = df["carry_paid"].mean() * 12
    est_manco_val = 4.0 * ann_mgmt + 0.2 * ann_carry

    kpis = pd.DataFrame({
        "Metric": [
            "Runway",
            "Break-even month",
            "Ending Company Cash ($)",
            "Ending Fund NAV ($)",
            "Annualized Mgmt Fees ($)",
            "Annualized Carry ($)",
            "Est. ManCo Value ($)"
        ],
        "Value": [
            runway_text,
            break_even,
            f"{df['company_cash'].iloc[-1]:,.0f}",
            f"{df['fund_nav'].iloc[-1]:,.0f}",
            f"{ann_mgmt:,.0f}",
            f"{ann_carry:,.0f}",
            f"{est_manco_val:,.0f}"
        ]
    })

    # Charts
    fund_nav = px.line(df, y="fund_nav", title="Fund NAV")
    aum_fee = go.Figure()
    aum_fee.add_trace(go.Bar(x=df.index, y=df["mgmt_fees"], name="Mgmt Fees (monthly)"))
    aum_fee.add_trace(go.Scatter(x=df.index, y=df["fund_aum"], mode="lines", name="AUM"))
    aum_fee.update_layout(title="AUM & Management Fees")

    company_cash = px.line(df, y="company_cash", title="Company Cash")
    runway_burn = go.Figure()
    runway_burn.add_trace(go.Bar(x=df.index, y=df["company_inflow"], name="Inflow"))
    runway_burn.add_trace(go.Bar(x=df.index, y=-df["monthly_burn"], name="Burn"))
    runway_burn.update_layout(barmode="relative", title="Company Inflow vs Burn")

    charts = dict(
        fund_nav=fund_nav,
        aum_fee_revenue=aum_fee,
        company_cash=company_cash,
        runway_burn=runway_burn
    )
    return charts, kpis

def run_mc(p, n_paths=250, seed=42):
    rng = np.random.default_rng(seed)
    paths_nav = []
    paths_cash = []
    be_months = []
    for i in range(n_paths):
        df = run_simulation(p, seed=int(rng.integers(0, 1_000_000)))
        paths_nav.append(df["fund_nav"].values)
        paths_cash.append(df["company_cash"].values)
        # break-even month
        be_idx = np.where(df["company_inflow"].values >= df["monthly_burn"].values)[0]
        be_months.append(be_idx[0] if len(be_idx) > 0 else np.nan)

    nav_arr = np.stack(paths_nav)
    cash_arr = np.stack(paths_cash)
    idx = df.index

    def pct_bounds(arr):
        lo = np.nanpercentile(arr, 10, axis=0)
        md = np.nanpercentile(arr, 50, axis=0)
        hi = np.nanpercentile(arr, 90, axis=0)
        return lo, md, hi

    nav_lo, nav_md, nav_hi = pct_bounds(nav_arr)
    cs_lo, cs_md, cs_hi = pct_bounds(cash_arr)

    fn = pd.DataFrame({"date": idx, "P10": nav_lo, "P50": nav_md, "P90": nav_hi}).set_index("date")
    cc = pd.DataFrame({"date": idx, "P10": cs_lo, "P50": cs_md, "P90": cs_hi}).set_index("date")

    fund_nav_plot = go.Figure()
    fund_nav_plot.add_trace(go.Scatter(x=fn.index, y=fn["P50"], name="NAV P50"))
    fund_nav_plot.add_trace(go.Scatter(x=fn.index, y=fn["P90"], name="NAV P90", line=dict(dash="dot")))
    fund_nav_plot.add_trace(go.Scatter(x=fn.index, y=fn["P10"], name="NAV P10", line=dict(dash="dot")))
    fund_nav_plot.update_layout(title="Fund NAV — Monte Carlo (P10/P50/P90)")

    company_cash_plot = go.Figure()
    company_cash_plot.add_trace(go.Scatter(x=cc.index, y=cc["P50"], name="Cash P50"))
    company_cash_plot.add_trace(go.Scatter(x=cc.index, y=cc["P90"], name="Cash P90", line=dict(dash="dot")))
    company_cash_plot.add_trace(go.Scatter(x=cc.index, y=cc["P10"], name="Cash P10", line=dict(dash="dot")))
    company_cash_plot.update_layout(title="Company Cash — Monte Carlo (P10/P50/P90)")

    be_series = pd.Series(be_months, name="BreakEvenMonth")
    break_even_hist = px.histogram(be_series.dropna(), nbins=20, title="Break-even Month Distribution")

    summary = pd.DataFrame({
        "Metric": ["P(Break-even ≤ 12m)", "P(Break-even ≤ 24m)", "Median Break-even (m)"],
        "Value": [
            f"{(be_series.dropna() <= 12).mean():.1%}",
            f"{(be_series.dropna() <= 24).mean():.1%}",
            f"{be_series.median():.1f}"
        ]
    })

    return dict(
        fund_nav_plot=fund_nav_plot,
        company_cash_plot=company_cash_plot,
        break_even_hist=break_even_hist,
        summary_table=summary
    )
    
# --- Sustainability grid helpers ---------------------------------------------

def _mc_prob_for_setting(p_base, n_paths=250, metric="no_cashout", be_months=24, seed=1234):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n_paths):
        df = run_simulation(p_base, seed=int(rng.integers(0, 1_000_000_000)))
        if metric == "no_cashout":
            cond = (df["cum_cash"].min() > 0)
        else:  # break-even within threshold months
            be_idx = np.where(df["company_inflow"].values >= df["monthly_burn"].values)[0]
            cond = (len(be_idx) > 0 and be_idx[0] <= be_months)
        ok += 1 if cond else 0
    return ok / n_paths


def sustainability_grid(p, aum_values, headcount_values, n_paths=250, metric="no_cashout", be_months=24, seed=123):
    """
    Returns a (len(headcount_values) x len(aum_values)) array of probabilities.
    Rows = headcount, Cols = AUM.
    """
    probs = np.zeros((len(headcount_values), len(aum_values)), dtype=float)
    for i, hc in enumerate(headcount_values):
        for j, aum0 in enumerate(aum_values):
            p2 = dict(p)
            p2["headcount"] = int(hc)
            p2["seed_fund_aum"] = float(aum0)
            probs[i, j] = _mc_prob_for_setting(
                p2, n_paths=n_paths, metric=metric, be_months=be_months,
                seed=seed + i * 10_000 + j
            )
    return probs

