# finance/params.py
def default_params():
    return dict(
        monthly_steps=36,
        founders_equity=0.60,
        employee_equity=0.20,
        investor_equity=0.20,
        headcount=7,
        avg_salary=220_000,  # fully-loaded / yr
        llm_cost=3_000,      # per month
        cloud_cost=1_500,    # per month
        other_opex=2_000,    # per month
        seed_company_cash=1_000_000,
        seed_fund_aum=5_000_000,
        mgmt_fee_pct=0.01,   # 1% annual
        carry_pct=0.20,      # 20% performance fee
        equity_only_investors=True,  # if True: investors get equity only (no fee share)
        mgmt_fee_paid_monthly=True,
        carry_high_water_mark=True,
        reinvest_carry=True,
        reinvest_mgmt_fee=True,
        hurdle_rate=0.00,
        target_ann_return=0.15,
        target_ann_vol=0.12,
        rf_rate=0.02,
        growth_new_aum=0.00,
        platform_take_on_ext_strats=False,
        start_date=None
    )
