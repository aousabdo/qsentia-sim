"""Run a single deterministic simulation and export plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance.cli_params import args_to_params, build_parser
from finance.sim import run_simulation, summarize_results


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Run a single-path fund/company simulation and save key plots.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/single",
        help="Directory where plots and CSV outputs will be saved (default: outputs/single).",
    )
    args = parser.parse_args(argv)

    params = args_to_params(args)
    df = run_simulation(params, seed=args.seed)
    charts, kpis = summarize_results(df, params)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "simulation.csv", index=True)
    kpis.to_csv(output_dir / "kpis.csv", index=False)

    def save_plot(fig, stem: str) -> None:
        html_path = output_dir / f"{stem}.html"
        png_path = output_dir / f"{stem}.png"
        pio.write_html(fig, html_path, include_plotlyjs="cdn")
        try:
            pio.write_image(fig, png_path, format="png", scale=2)
        except ValueError as err:
            print(f"[warn] Skipped PNG export for {stem}: {err}", file=sys.stderr)

    save_plot(charts["fund_nav"], "fund_nav")
    save_plot(charts["company_cash"], "company_cash")

    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
