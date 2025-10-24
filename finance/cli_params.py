"""Helpers for CLI scripts to collect simulation parameters."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from finance.params import default_params


def _add_toggle(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.set_defaults(**{name: default})
    if default:
        flag = f"--no-{name.replace('_', '-')}"
        parser.add_argument(
            flag,
            dest=name,
            action="store_false",
            help=f"Disable {help_text} (default: enabled)",
        )
    else:
        flag = f"--{name.replace('_', '-')}"
        parser.add_argument(
            flag,
            dest=name,
            action="store_true",
            help=f"Enable {help_text} (default: disabled)",
        )


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    defaults = default_params()

    p.add_argument("--horizon-years", type=int, default=defaults["monthly_steps"] // 12,
                   help="Simulation horizon in years (default: 5).")
    p.add_argument("--founders-equity", type=float, default=defaults["founders_equity"] * 100,
                   help="Founders equity percentage (default: 60).")
    p.add_argument("--employee-equity", type=float, default=defaults["employee_equity"] * 100,
                   help="Employee pool equity percentage (default: 20).")
    p.add_argument("--investor-equity", type=float, default=None,
                   help="Investor equity percentage (default: 100 - founders - employee).")
    p.add_argument("--headcount", type=int, default=defaults["headcount"],
                   help="Initial headcount (default: 7).")
    p.add_argument("--avg-salary", type=float, default=defaults["avg_salary"],
                   help="Average fully-loaded salary per FTE in USD/year (default: 220000).")
    p.add_argument("--llm-cost", type=float, default=defaults["llm_cost"],
                   help="Monthly LLM/Data/API cost in USD (default: 3000).")
    p.add_argument("--cloud-cost", type=float, default=defaults["cloud_cost"],
                   help="Monthly cloud/infra cost in USD (default: 1500).")
    p.add_argument("--other-opex", type=float, default=defaults["other_opex"],
                   help="Other monthly operating expenses in USD (default: 2000).")
    p.add_argument("--seed-company-cash", type=float, default=defaults["seed_company_cash"],
                   help="Initial company cash in USD (default: 1000000).")
    p.add_argument("--seed-fund-aum", type=float, default=defaults["seed_fund_aum"],
                   help="Initial seed fund AUM in USD (default: 5000000).")
    p.add_argument("--mgmt-fee-pct", type=float, default=defaults["mgmt_fee_pct"] * 100,
                   help="Management fee percentage (annual, default: 1).")
    p.add_argument("--carry-pct", type=float, default=defaults["carry_pct"] * 100,
                   help="Performance fee / carry percentage (default: 20).")
    _add_toggle(p, "equity_only_investors", defaults["equity_only_investors"], "equity-only investors")
    _add_toggle(p, "mgmt_fee_paid_monthly", defaults["mgmt_fee_paid_monthly"], "monthly management fee collection")
    _add_toggle(p, "carry_high_water_mark", defaults["carry_high_water_mark"], "high-water mark for carry")
    _add_toggle(p, "reinvest_carry", defaults["reinvest_carry"], "carry reinvestment into company cash")
    _add_toggle(p, "reinvest_mgmt_fee", defaults["reinvest_mgmt_fee"], "management fee reinvestment")
    p.add_argument("--hurdle-rate", type=float, default=defaults["hurdle_rate"] * 100,
                   help="Annual hurdle rate percentage (default: 0).")
    p.add_argument("--target-ann-return", type=float, default=defaults["target_ann_return"] * 100,
                   help="Target annual return percentage (default: 15).")
    p.add_argument("--target-ann-vol", type=float, default=defaults["target_ann_vol"] * 100,
                   help="Annual volatility percentage (default: 12).")
    p.add_argument("--rf-rate", type=float, default=defaults["rf_rate"] * 100,
                   help="Risk-free rate percentage (default: 2).")
    p.add_argument("--growth-new-aum", type=float, default=defaults["growth_new_aum"] * 100,
                   help="Organic AUM growth percentage (annual, default: 0).")
    _add_toggle(p, "platform_take_on_ext_strats", defaults["platform_take_on_ext_strats"],
                "platform external strategies uplift")
    p.add_argument("--start-date", type=str, default=defaults.get("start_date"),
                   help="Optional ISO start date (default: month-end today).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for simulation (default: 42).")

    return p


def args_to_params(args: argparse.Namespace) -> Dict[str, Any]:
    params = default_params()

    params["monthly_steps"] = int(args.horizon_years) * 12
    params["founders_equity"] = float(args.founders_equity) / 100.0
    params["employee_equity"] = float(args.employee_equity) / 100.0

    if args.investor_equity is None:
        inv_pct = max(0.0, 100.0 - args.founders_equity - args.employee_equity)
    else:
        inv_pct = float(args.investor_equity)
    params["investor_equity"] = inv_pct / 100.0

    params["headcount"] = int(args.headcount)
    params["avg_salary"] = float(args.avg_salary)
    params["llm_cost"] = float(args.llm_cost)
    params["cloud_cost"] = float(args.cloud_cost)
    params["other_opex"] = float(args.other_opex)
    params["seed_company_cash"] = float(args.seed_company_cash)
    params["seed_fund_aum"] = float(args.seed_fund_aum)
    params["mgmt_fee_pct"] = float(args.mgmt_fee_pct) / 100.0
    params["carry_pct"] = float(args.carry_pct) / 100.0

    params["equity_only_investors"] = bool(args.equity_only_investors)
    params["mgmt_fee_paid_monthly"] = bool(args.mgmt_fee_paid_monthly)
    params["carry_high_water_mark"] = bool(args.carry_high_water_mark)
    params["reinvest_carry"] = bool(args.reinvest_carry)
    params["reinvest_mgmt_fee"] = bool(args.reinvest_mgmt_fee)
    params["hurdle_rate"] = float(args.hurdle_rate) / 100.0
    params["target_ann_return"] = float(args.target_ann_return) / 100.0
    params["target_ann_vol"] = float(args.target_ann_vol) / 100.0
    params["rf_rate"] = float(args.rf_rate) / 100.0
    params["growth_new_aum"] = float(args.growth_new_aum) / 100.0
    params["platform_take_on_ext_strats"] = bool(args.platform_take_on_ext_strats)
    params["start_date"] = args.start_date

    return params
