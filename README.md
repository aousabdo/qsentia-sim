# QSentia — Equity & Fund Economics Simulator

Interactive Streamlit app to model company runway, fund fees, carry, and equity splits under different team, cost, and performance assumptions.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py

## CLI helpers

Two utility scripts mirror the Streamlit controls and export the core plots locally:

```bash
# Single-path deterministic run (saves HTML, PNG, CSV to outputs/single)
python scripts/run_single_path.py --help

# Monte Carlo aggregate run (saves HTML, PNG, CSV to outputs/monte_carlo)
python scripts/run_monte_carlo.py --help
```

Both scripts accept the same parameters as the app (e.g. `--horizon-years`, `--founders-equity`,
`--mgmt-fee-pct`, etc.). Run with `--help` to see the full list.

Example with custom inputs:

```bash
python scripts/run_single_path.py \
  --horizon-years 5 \
  --founders-equity 55 --employee-equity 25 \
  --headcount 10 --avg-salary 250000 \
  --llm-cost 4000 --cloud-cost 2000 --other-opex 2500 \
  --seed-company-cash 1500000 --seed-fund-aum 8000000 \
  --mgmt-fee-pct 1.5 --carry-pct 18 \
  --no-reinvest-carry --no-reinvest-mgmt-fee
```

Boolean switches mirror the app’s toggles: prefix with `--no-` to disable defaults (e.g. `--no-reinvest-carry`),
or use the plain flag to enable features that are off by default (e.g. `--platform-take-on-ext-strats`).
