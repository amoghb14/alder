"""Walk-forward evaluation CLI for the ALDER models.

Usage:
    python model_eval.py            # default basket
    python model_eval.py NVDA AAPL

Every score is produced chronologically: the models are refit on past bars only and
then asked to predict the next five sessions. Naive baselines are printed next to the
model so a reader can see whether the model actually beats a rule of thumb.
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import models

DEFAULT = ["NVDA", "AAPL", "MSFT"]


def main(symbols: list[str]) -> int:
    header = (
        f"{'ticker':<7}{'OOS':>5}{'retMAE':>9}{'retR2':>8}{'priceMAE':>10}"
        f"{'volR2':>8}{'up/dn':>7}{'naive':>7}{'hit':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for symbol in symbols:
        try:
            result = models.evaluate(symbol)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"{symbol:<7} failed: {exc}")
            continue
        researcher, risk, strategist = result["researcher"], result["risk"], result["strategist"]
        print(
            f"{symbol:<7}{result['test_rows']:>5}{researcher['mae_bps']:>8.0f}b"
            f"{researcher['r2']:>8.3f}{researcher['price_mae']:>9.2f}"
            f"{risk['r2']:>8.3f}{strategist['updown_accuracy']:>6.1f}%"
            f"{strategist['always_up']:>6.1f}%{researcher['hit_rate']:>6.1f}%"
        )
        rows.append((symbol, strategist["updown_accuracy"], strategist["always_up"]))

    if rows:
        edge = sum(up - naive for _, up, naive in rows) / len(rows)
        print(f"\nMean directional accuracy {sum(up for _, up, _ in rows) / len(rows):.1f}% "
              f"vs always-up {sum(naive for _, _, naive in rows) / len(rows):.1f}% "
              f"(edge {edge:+.1f} pp)")
        print("Return R² near zero is expected: daily-to-weekly equity returns are mostly noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or DEFAULT))
