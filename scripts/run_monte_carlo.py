"""Run Monte Carlo simulations and export aggregate plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance.cli_params import args_to_params, build_parser
from finance.sim import run_mc


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Run Monte Carlo fund/company simulations and save aggregate plots.")
    parser.add_argument(
        "--mc-paths",
        type=int,
        default=250,
        help="Number of Monte Carlo paths to simulate (default: 250).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/monte_carlo",
        help="Directory where plots and summary outputs will be saved (default: outputs/monte_carlo).",
    )
    args = parser.parse_args(argv)

    params = args_to_params(args)
    results = run_mc(params, n_paths=int(args.mc_paths), seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_plot(fig, stem: str) -> None:
        html_path = output_dir / f"{stem}.html"
        png_path = output_dir / f"{stem}.png"
        pio.write_html(fig, html_path, include_plotlyjs="cdn")
        try:
            pio.write_image(fig, png_path, format="png", scale=2)
        except ValueError as err:
            print(f"[warn] Skipped PNG export for {stem}: {err}", file=sys.stderr)

    save_plot(results["fund_nav_plot"], "fund_nav_mc")
    save_plot(results["company_cash_plot"], "company_cash_mc")
    save_plot(results["break_even_hist"], "break_even_hist")

    results["summary_table"].to_csv(output_dir / "summary_table.csv", index=False)

    print(f"Saved Monte Carlo outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
