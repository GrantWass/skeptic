"""
Generates research reports from threshold optimization results.

Outputs:
  - data/reports/optimal_params.csv      — best (buy, sell) per asset
  - data/reports/asset_ranking.csv       — assets ranked at fixed thresholds
  - data/reports/report_summary.md       — TL;DR + Methodology + Asset Ranking
  - data/reports/report_params.md        — Optimal Parameters per Asset
  - data/reports/report_profit.md        — Estimated Profit
  - data/reports/report_time_of_day.md   — Volatility + Edge by Time of Day
  - data/reports/report_prior_resolution.md — Strategy by Prior Window Resolution
  - data/reports/report_30pct.md            — ≥30% Fill Rate Analysis
  - data/reports/report_recommendation.md  — Recommendation
"""
import glob
import os
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from skeptic import config
from skeptic.research.analyzer import (ThresholdResult, simulate, optimize_thresholds_sided,
                                        optimize_thresholds_sided_3d, group_by_prev_resolution,
                                        neighborhood_robustness, best_neighborhood_params_min_fill_rate)

log = logging.getLogger(__name__)


def _ensure_reports_dir() -> None:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)


def save_optimal_params(
    per_asset_best: dict[str, dict],
    all_best: dict,
) -> str:
    """Save optimal params table to CSV and return path."""
    _ensure_reports_dir()
    rows = []
    for asset, params in per_asset_best.items():
        rows.append({"asset": asset, **params})
    df = pd.DataFrame(rows)
    path = os.path.join(config.REPORTS_DIR, "optimal_params.csv")
    df.to_csv(path, index=False)
    log.info("Saved optimal params to %s", path)
    return path


def save_asset_ranking(df: pd.DataFrame) -> str:
    _ensure_reports_dir()
    path = os.path.join(config.REPORTS_DIR, "asset_ranking.csv")
    df.to_csv(path, index=False)
    log.info("Saved asset ranking to %s", path)
    return path


def save_full_grid(asset: str, df: pd.DataFrame) -> str:
    _ensure_reports_dir()
    path = os.path.join(config.REPORTS_DIR, f"grid_{asset.lower()}.csv")
    df.head(200).to_csv(path, index=False)
    log.info("Saved grid for %s to %s", asset, path)
    return path


def cache_grid(asset: str, df: pd.DataFrame, args_key: str) -> None:
    """Persist the full grid DataFrame and the args key used to produce it."""
    _ensure_reports_dir()
    pkl_path  = os.path.join(config.REPORTS_DIR, f"grid_cache_{asset.lower()}.pkl")
    args_path = os.path.join(config.REPORTS_DIR, f"grid_cache_{asset.lower()}_args.txt")
    df.to_pickle(pkl_path)
    with open(args_path, "w") as f:
        f.write(args_key)
    log.info("Cached grid for %s (%d rows)", asset, len(df))


def load_cached_grid(asset: str, args_key: str) -> "pd.DataFrame | None":
    """
    Return the cached grid if:
      - cache file exists and args_key matches
      - cache is newer than the most recent price CSV

    Returns None on any cache miss so the caller falls back to a full grid search.
    """
    pkl_path  = os.path.join(config.REPORTS_DIR, f"grid_cache_{asset.lower()}.pkl")
    args_path = os.path.join(config.REPORTS_DIR, f"grid_cache_{asset.lower()}_args.txt")

    if not os.path.exists(pkl_path) or not os.path.exists(args_path):
        return None

    with open(args_path) as f:
        if f.read().strip() != args_key.strip():
            log.info("Cache miss for %s — args changed", asset)
            return None

    cache_mtime = os.path.getmtime(pkl_path)
    price_files = glob.glob(os.path.join("data", "prices", "*.csv"))
    if not price_files:
        return None
    newest_price = max(os.path.getmtime(p) for p in price_files)
    if newest_price > cache_mtime:
        log.info("Cache miss for %s — price data is newer", asset)
        return None

    df = pd.read_pickle(pkl_path)
    log.info("Cache hit for %s (%d rows)", asset, len(df))
    return df


def _sided_cache_paths(asset: str, group: str, side: str) -> tuple[str, str]:
    slug = f"{asset.lower()}_{group}_{side}"
    return (
        os.path.join(config.REPORTS_DIR, f"sided_cache_{slug}.pkl"),
        os.path.join(config.REPORTS_DIR, f"sided_cache_{slug}_args.txt"),
    )


def _cache_sided_grid(asset: str, group: str, side: str, df: pd.DataFrame, args_key: str) -> None:
    _ensure_reports_dir()
    pkl_path, args_path = _sided_cache_paths(asset, group, side)
    df.to_pickle(pkl_path)
    with open(args_path, "w") as f:
        f.write(args_key)
    log.info("Cached sided grid %s/%s/%s (%d rows)", asset, group, side, len(df))


def _load_cached_sided_grid(asset: str, group: str, side: str, args_key: str) -> "pd.DataFrame | None":
    pkl_path, args_path = _sided_cache_paths(asset, group, side)
    if not os.path.exists(pkl_path) or not os.path.exists(args_path):
        return None
    with open(args_path) as f:
        if f.read().strip() != args_key.strip():
            log.info("Sided cache miss %s/%s/%s — args changed", asset, group, side)
            return None
    cache_mtime = os.path.getmtime(pkl_path)
    price_files = glob.glob(os.path.join("data", "prices", "*.csv"))
    if not price_files:
        return None
    if max(os.path.getmtime(p) for p in price_files) > cache_mtime:
        log.info("Sided cache miss %s/%s/%s — price data is newer", asset, group, side)
        return None
    df = pd.read_pickle(pkl_path)
    log.info("Sided cache hit %s/%s/%s (%d rows)", asset, group, side, len(df))
    return df


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "—"

def _edge(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

def _price(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

def _ratio(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

def _compute_hourly_volatility(
    all_sessions: dict,
    fill_windows: list[int],
    bucket_hours: int = 2,
) -> list[str]:
    """
    For each fill window, compute mean price volatility (max-min) per N-hour bucket.
    Returns markdown lines for the section.
    """
    CT_OFFSET = -6  # UTC-6 = Central Standard Time

    lines: list[str] = []

    assets = list(all_sessions.keys())

    for fw in fill_windows:
        rows = []
        for asset, sessions in all_sessions.items():
            for s in sessions:
                cutoff = s.window_start_ts + fw
                up_prices = [p for ts, p in s.up_trades_all if ts <= cutoff]
                dn_prices = [p for ts, p in s.down_trades_all if ts <= cutoff]
                all_prices = up_prices + dn_prices
                if len(all_prices) < 2:
                    continue
                volatility = max(all_prices) - min(all_prices)
                utc_hour = datetime.fromtimestamp(s.window_start_ts, tz=timezone.utc).hour
                utc_bucket = (utc_hour // bucket_hours) * bucket_hours
                ct_bucket = (utc_bucket + CT_OFFSET) % 24
                rows.append({
                    "utc_bucket": utc_bucket,
                    "ct_bucket": ct_bucket,
                    "asset": asset,
                    "volatility": volatility,
                })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        # Overall volatility per bucket
        overall = (
            df.groupby("utc_bucket")
            .agg(n_sessions=("volatility", "count"), mean_vol=("volatility", "mean"),
                 ct_bucket=("ct_bucket", "first"))
            .reset_index()
            .sort_values("mean_vol", ascending=False)
        )

        # Per-asset mean volatility per bucket
        per_asset = (
            df.groupby(["utc_bucket", "asset"])["volatility"]
            .mean()
            .unstack()
            .fillna(0.0)
        )
        # Ensure all assets are present as columns
        for a in assets:
            if a not in per_asset.columns:
                per_asset[a] = 0.0
        per_asset = per_asset[assets]

        asset_cols = " | ".join(assets)
        asset_sep = " | ".join(["-------"] * len(assets))

        lines.append(f"\n#### Fill Window: {fw}s\n")
        lines.append(f"| UTC Period | CT Period | N Sessions | Mean Vol | {asset_cols} |")
        lines.append(f"|------------|-----------|------------|----------|{asset_sep}|")
        for _, row in overall.iterrows():
            u = int(row["utc_bucket"])
            c = int(row["ct_bucket"])
            asset_vals = " | ".join(
                f"{per_asset.loc[u, a]:.4f}" if u in per_asset.index else "—"
                for a in assets
            )
            lines.append(
                f"| {u:02d}–{(u + bucket_hours) % 24:02d} UTC | "
                f"{c:02d}–{(c + bucket_hours) % 24:02d} CT | "
                f"{int(row['n_sessions'])} | "
                f"{row['mean_vol']:.4f} | "
                f"{asset_vals} |"
            )

    return lines


def _edge_by_hour(
    all_sessions: dict,
    best_nb: dict[str, dict],
    bucket_hours: int = 3,
) -> list[str]:
    """
    For each time bucket, run simulate() for each asset using its own best
    neighborhood buy/sell/fill_window. Returns markdown lines.
    """
    CT_OFFSET = -6
    assets = [a for a in all_sessions if best_nb.get(a)]
    if not assets:
        return []

    # Build {utc_bucket: {asset: [sessions]}}
    buckets: dict[int, dict[str, list]] = {}
    for asset in assets:
        for s in all_sessions[asset]:
            utc_hour = datetime.fromtimestamp(s.window_start_ts, tz=timezone.utc).hour
            b = (utc_hour // bucket_hours) * bucket_hours
            buckets.setdefault(b, {}).setdefault(asset, []).append(s)

    if not buckets:
        return []

    # Header shows each asset's params so reader knows what was used
    param_notes = "  ".join(
        f"{a}: buy={best_nb[a].get('buy'):.2f} sell={best_nb[a].get('sell'):.2f} fw={best_nb[a].get('fill_window') or '?'}s"
        for a in assets
    )

    rows = []
    for utc_bucket in sorted(buckets):
        ct_bucket = (utc_bucket + CT_OFFSET) % 24
        row: dict = {"utc_bucket": utc_bucket, "ct_bucket": ct_bucket, "n_sessions": 0}
        for asset in assets:
            nb  = best_nb[asset]
            nb_buy  = nb.get("buy")
            nb_sell = nb.get("sell")
            nb_fw   = nb.get("fill_window") or 60
            sessions_in_bucket = buckets[utc_bucket].get(asset, [])
            n = len(sessions_in_bucket)
            row["n_sessions"] += n
            if n == 0 or not nb_buy or not nb_sell:
                row[asset] = None
            else:
                r = simulate(sessions_in_bucket, nb_buy, nb_sell, fill_window=int(nb_fw))
                row[asset] = r.edge_per_session
        rows.append(row)

    asset_cols = " | ".join(assets)
    asset_sep  = " | ".join(["-------"] * len(assets))

    lines = [
        f"_Params used — {param_notes}_",
        "",
        f"| UTC Period | CT Period | N Sessions | {asset_cols} | Profitable |",
        f"|------------|-----------|------------|{asset_sep}|------------|",
    ]
    for row in rows:
        u = row["utc_bucket"]
        c = row["ct_bucket"]
        asset_vals = " | ".join(
            f"{row[a]:+.4f}" if row[a] is not None else "—"
            for a in assets
        )
        n_pos   = sum(1 for a in assets if row[a] is not None and row[a] > 0)
        n_valid = sum(1 for a in assets if row[a] is not None)
        lines.append(
            f"| {u:02d}–{(u + bucket_hours) % 24:02d} UTC | "
            f"{c:02d}–{(c + bucket_hours) % 24:02d} CT | "
            f"{row['n_sessions']} | "
            f"{asset_vals} | "
            f"{n_pos}/{n_valid} |"
        )

    return lines


def _load_cached_grids(assets: list[str]) -> dict[str, "pd.DataFrame"]:
    """Load full grid DataFrames from pickle cache. Returns empty dict if not found."""
    grids: dict[str, pd.DataFrame] = {}
    for asset in assets:
        path = os.path.join(config.REPORTS_DIR, f"grid_cache_{asset.lower()}.pkl")
        if os.path.exists(path):
            try:
                grids[asset] = pd.read_pickle(path)
            except Exception as e:
                log.warning("Could not load cached grid for %s: %s", asset, e)
    return grids


def _prev_resolution_section(
    all_sessions: dict,
    buy_range: tuple[float, float],
    sell_range: tuple[float, float],
    step: float,
    fill_window_range: tuple[int, int] = (10, 60),
    fill_window_step: int = 10,
    min_fill_rate: float = 0.0,
) -> list[str]:
    """
    For each asset, split sessions by previous window resolution (UP or DOWN),
    then grid-search the best strategy for each side (bet UP only, DOWN only, or both).
    Appends an analysis section summarising cross-asset patterns.
    Returns markdown lines.
    """
    SIDES = [("both", "Both (original)"), ("up", "Bet UP only"), ("down", "Bet DOWN only")]
    lines: list[str] = []

    _args_key = (
        f"buy={buy_range} sell={sell_range} step={step} "
        f"fw={fill_window_range},{fill_window_step} mfr={min_fill_rate}"
    )

    def _get_sided_df(asset: str, group: str, sessions_for_group: list) -> pd.DataFrame:
        """Load from cache or run 3D grid search, then save."""
        cached = _load_cached_sided_grid(asset, group, side, _args_key)
        if cached is not None:
            return cached
        result = optimize_thresholds_sided_3d(
            sessions_for_group, buy_range, sell_range, step,
            fill_window_range, fill_window_step, side=side,
        )
        _cache_sided_grid(asset, group, side, result, _args_key)
        return result

    # ── Regardless of prior resolution ───────────────────────────────────────
    lines += [
        "## Regardless of Prior Resolution",
        "",
        "> All sessions included — no prior resolution filter.",
        "> Compares betting both sides, UP only, or DOWN only using the same grid search.",
        "",
        "| Asset | Bet Side | N Sessions | Buy | Sell | Fill Rate | Sell Hit% | Edge/Session | NB Mean Edge | % NB Positive |",
        "|-------|----------|------------|-----|------|-----------|-----------|--------------|--------------|---------------|",
    ]
    for asset, sessions in all_sessions.items():
        for side, side_label in SIDES:
            df = _get_sided_df(asset, "all", sessions)
            if min_fill_rate > 0:
                df = pd.DataFrame(df[df["fill_rate"] >= min_fill_rate - 1e-9]).reset_index(drop=True)
            if df.empty:
                lines.append(f"| {asset} | {side_label} | {len(sessions)} | — | — | — | — | — | — | — |")
                continue
            best = df.iloc[0]
            nb = neighborhood_robustness(df, best.to_dict())
            nb_mean = _edge(nb.get("neighbor_mean_edge")) if nb else "—"
            nb_pct  = _pct(nb.get("pct_positive"))       if nb else "—"
            lines.append(
                f"| {asset} | {side_label} | {len(sessions)} | {_price(best['buy'])} | {_price(best['sell'])} "
                f"| {_pct(best['fill_rate'])} | {_pct(best['sell_hit_rate'])} "
                f"| {_edge(best['edge_per_session'])} | {nb_mean} | {nb_pct} |"
            )
    lines += ["", "---", ""]

    # Collect results for cross-asset analysis
    # results[asset] = {"prev_up": {side: edge}, "prev_down": {side: edge},
    #                   "n_prev_up": int, "n_prev_down": int}
    results: dict[str, dict] = {}

    for asset, sessions in all_sessions.items():
        prev_up, prev_down, n_disc = group_by_prev_resolution(sessions)
        total = len(prev_up) + len(prev_down)

        lines += [
            f"\n### {asset}",
            f"",
            f"_{len(sessions)} total sessions — {total} with confirmed prior resolution "
            f"({len(prev_up)} prev UP, {len(prev_down)} prev DOWN, {n_disc} discarded)_",
            "",
        ]

        if total == 0:
            lines.append("_Insufficient data._\n")
            continue

        results[asset] = {"prev_up": {}, "prev_down": {}, "n_prev_up": len(prev_up), "n_prev_down": len(prev_down)}

        for group_key, label, group_sessions in [
            ("prev_up",   "Previous resolution: UP",   prev_up),
            ("prev_down", "Previous resolution: DOWN",  prev_down),
        ]:
            n = len(group_sessions)
            lines += [f"**{label}** ({n} sessions)\n"]

            if n < 5:
                lines.append("_Too few sessions to optimize._\n")
                continue

            lines += [
                "| Bet Side | Buy | Sell | Fill Rate | Sell Hit% | Edge/Session | NB Mean Edge | % NB Positive |",
                "|----------|-----|------|-----------|-----------|--------------|--------------|---------------|",
            ]
            for side, side_label in SIDES:
                df = _get_sided_df(asset, group_key, group_sessions)
                if min_fill_rate > 0:
                    df = pd.DataFrame(df[df["fill_rate"] >= min_fill_rate - 1e-9]).reset_index(drop=True)
                if df.empty:
                    lines.append(f"| {side_label} | — | — | — | — | — | — | — |")
                    continue
                best = df.iloc[0]
                results[asset][group_key][side] = float(best["edge_per_session"])
                nb = neighborhood_robustness(df, best.to_dict())
                nb_mean = _edge(nb.get("neighbor_mean_edge")) if nb else "—"
                nb_pct  = _pct(nb.get("pct_positive"))       if nb else "—"
                lines.append(
                    f"| {side_label} | {_price(best['buy'])} | {_price(best['sell'])} "
                    f"| {_pct(best['fill_rate'])} | {_pct(best['sell_hit_rate'])} "
                    f"| {_edge(best['edge_per_session'])} | {nb_mean} | {nb_pct} |"
                )
            lines.append("")

    # ── Cross-asset analysis ──────────────────────────────────────────────────
    if results:
        lines += [
            "",
            "---",
            "",
            "## Analysis",
            "",
        ]

        # For each (prior_resolution, group) determine: which side won, and what does that mean?
        # After prev UP:  betting UP = momentum,       betting DOWN = mean reversion
        # After prev DOWN: betting UP = mean reversion, betting DOWN = momentum
        def _classify(group_key: str, winning_side: str) -> str:
            if group_key == "prev_up":
                return "momentum" if winning_side == "up" else ("mean reversion" if winning_side == "down" else "no edge")
            else:
                return "mean reversion" if winning_side == "up" else ("momentum" if winning_side == "down" else "no edge")

        # Summary table: after prev UP
        lines += [
            "### After Previous Resolution: UP",
            "",
            "| Asset | N Sessions | Best Side | Edge/Session | Pattern |",
            "|-------|------------|-----------|--------------|---------|",
        ]
        up_patterns: list[str] = []
        for asset, r in results.items():
            edges = r["prev_up"]
            if not edges:
                lines.append(f"| {asset} | {r['n_prev_up']} | — | — | — |")
                continue
            best_side = max(edges, key=edges.get)
            best_edge = edges[best_side]
            pattern = _classify("prev_up", best_side)
            up_patterns.append(pattern)
            side_label = {"both": "Both", "up": "Bet UP", "down": "Bet DOWN"}[best_side]
            lines.append(f"| {asset} | {r['n_prev_up']} | {side_label} | {_edge(best_edge)} | {pattern} |")

        # Summary table: after prev DOWN
        lines += [
            "",
            "### After Previous Resolution: DOWN",
            "",
            "| Asset | N Sessions | Best Side | Edge/Session | Pattern |",
            "|-------|------------|-----------|--------------|---------|",
        ]
        down_patterns: list[str] = []
        for asset, r in results.items():
            edges = r["prev_down"]
            if not edges:
                lines.append(f"| {asset} | {r['n_prev_down']} | — | — | — |")
                continue
            best_side = max(edges, key=edges.get)
            best_edge = edges[best_side]
            pattern = _classify("prev_down", best_side)
            down_patterns.append(pattern)
            side_label = {"both": "Both", "up": "Bet UP", "down": "Bet DOWN"}[best_side]
            lines.append(f"| {asset} | {r['n_prev_down']} | {side_label} | {_edge(best_edge)} | {pattern} |")

        # Which prior is more profitable?
        lines += ["", "### Which Prior Resolution is More Profitable?", ""]
        lines += [
            "| Asset | Best Edge after UP | Best Edge after DOWN | Edge Advantage |",
            "|-------|-------------------|---------------------|----------------|",
        ]
        for asset, r in results.items():
            up_best   = max(r["prev_up"].values(),   default=None)
            down_best = max(r["prev_down"].values(), default=None)
            if up_best is None or down_best is None:
                lines.append(f"| {asset} | — | — | — |")
                continue
            adv = "prev UP better" if up_best > down_best else ("prev DOWN better" if down_best > up_best else "equal")
            lines.append(f"| {asset} | {_edge(up_best)} | {_edge(down_best)} | {adv} |")

        # Narrative conclusions
        n_momentum_after_up   = up_patterns.count("momentum")
        n_meanrev_after_up    = up_patterns.count("mean reversion")
        n_momentum_after_down = down_patterns.count("momentum")
        n_meanrev_after_down  = down_patterns.count("mean reversion")
        n_assets = len(results)

        lines += ["", "### Key Takeaways", ""]

        # After UP
        if n_momentum_after_up >= n_assets * 0.6:
            lines.append(
                f"- **After UP resolution — Momentum dominates ({n_momentum_after_up}/{n_assets} assets):** "
                f"When the prior window resolved UP, betting UP again outperforms. "
                f"Crypto markets exhibit short-term momentum — winners keep winning across 5-minute windows."
            )
        elif n_meanrev_after_up >= n_assets * 0.6:
            lines.append(
                f"- **After UP resolution — Mean reversion dominates ({n_meanrev_after_up}/{n_assets} assets):** "
                f"When the prior window resolved UP, fading it by betting DOWN outperforms. "
                f"Markets tend to oscillate rather than trend."
            )
        else:
            lines.append(
                f"- **After UP resolution — Mixed signal ({n_momentum_after_up} momentum, "
                f"{n_meanrev_after_up} mean reversion):** No consistent pattern across assets."
            )

        # After DOWN
        if n_meanrev_after_down >= n_assets * 0.6:
            lines.append(
                f"- **After DOWN resolution — Mean reversion dominates ({n_meanrev_after_down}/{n_assets} assets):** "
                f"When the prior window resolved DOWN, betting UP (fading the move) outperforms. "
                f"Combined with the after-UP result, this suggests an UP bias in the market — "
                f"UP is the preferred bet regardless of what just happened."
            )
        elif n_momentum_after_down >= n_assets * 0.6:
            lines.append(
                f"- **After DOWN resolution — Momentum dominates ({n_momentum_after_down}/{n_assets} assets):** "
                f"When the prior window resolved DOWN, betting DOWN again outperforms."
            )
        else:
            lines.append(
                f"- **After DOWN resolution — Mixed signal ({n_momentum_after_down} momentum, "
                f"{n_meanrev_after_down} mean reversion):** No consistent pattern across assets."
            )

        # UP bias check
        if n_momentum_after_up >= n_assets * 0.6 and n_meanrev_after_down >= n_assets * 0.6:
            lines.append(
                "- **Systematic UP bias detected:** Betting UP wins after both UP and DOWN resolutions "
                "across most assets. This likely reflects the inherent upward bias in crypto prices — "
                "markets expect assets to go up, so the UP side resolves correctly more often. "
                "Consider restricting your strategy to only bet UP regardless of prior resolution."
            )

        # Prior which is better
        prev_down_better = sum(
            1 for r in results.values()
            if r["prev_down"] and r["prev_up"]
            and max(r["prev_down"].values(), default=0) > max(r["prev_up"].values(), default=0)
        )
        prev_up_better = sum(
            1 for r in results.values()
            if r["prev_down"] and r["prev_up"]
            and max(r["prev_up"].values(), default=0) > max(r["prev_down"].values(), default=0)
        )
        if prev_down_better > prev_up_better:
            lines.append(
                f"- **Sessions after a DOWN resolution tend to be more profitable ({prev_down_better}/{n_assets} assets):** "
                f"More edge is available in windows that follow a DOWN outcome. "
                f"This may indicate that DOWN resolutions create more mispricing in the next window."
            )
        elif prev_up_better > prev_down_better:
            lines.append(
                f"- **Sessions after an UP resolution tend to be more profitable ({prev_up_better}/{n_assets} assets):** "
                f"More edge is available in windows that follow an UP outcome."
            )

        # Both strategy note
        both_suboptimal_up = sum(
            1 for r in results.values()
            if r["prev_up"] and r["prev_up"].get("both", -999) < max(r["prev_up"].values(), default=0)
        )
        both_suboptimal_down = sum(
            1 for r in results.values()
            if r["prev_down"] and r["prev_down"].get("both", -999) < max(r["prev_down"].values(), default=0)
        )
        if both_suboptimal_up + both_suboptimal_down > n_assets:
            lines.append(
                f"- **The 'bet both' strategy is suboptimal once you condition on prior resolution:** "
                f"A directional bet (UP only or DOWN only) outperforms the original strategy for most assets "
                f"in both prior-UP and prior-DOWN groups. Knowing the prior resolution is a useful signal."
            )

    return lines


def _report_30pct_section(
    all_sessions: dict,
    per_asset_best_30pct: dict[str, dict],
    buy_range: tuple[float, float],
    sell_range: tuple[float, float],
    step: float,
    fill_window_range: tuple[int, int] = (10, 60),
    fill_window_step: int = 10,
) -> list[str]:
    """Full ≥30% fill rate report: best params, best neighborhood, directional + prior resolution analysis."""
    MIN_FR = 0.30
    lines: list[str] = []

    # ── 1. Best params at ≥30% ───────────────────────────────────────────────
    lines += [
        "## Best Parameters at ≥30% Fill Rate",
        "",
        "> Only grid combinations with fill_rate ≥ 30% are eligible.",
        "",
        "| Asset | Buy | Sell | Fill Win | Fill Rate | Sell Hit% | Edge/Session |",
        "|-------|-----|------|----------|-----------|-----------|--------------|",
    ]
    for asset, p in per_asset_best_30pct.items():
        if not p:
            lines.append(f"| {asset} | — | — | — | — | — | _no combo meets 30%_ |")
            continue
        fw = f"{int(p['fill_window'])}s" if "fill_window" in p else "60s"
        lines.append(
            f"| {asset} | {_price(p.get('buy'))} | {_price(p.get('sell'))} | {fw} "
            f"| {_pct(p.get('fill_rate'))} | {_pct(p.get('sell_hit_rate'))} "
            f"| {_edge(p.get('edge_per_session'))} |"
        )

    # ── 2. Best neighborhood at ≥30% (from cached grids) ────────────────────
    grids = _load_cached_grids(list(all_sessions.keys()))
    if grids:
        lines += [
            "",
            "---",
            "",
            "## Best Neighborhood at ≥30% Fill Rate",
            "",
            "> Neighborhood search restricted to grid points with fill_rate ≥ 30%.",
            "",
            "| Asset | NB Buy | NB Sell | NB Fill Win | NB Mean Edge | % NB Positive | Peak vs NB |",
            "|-------|--------|---------|-------------|--------------|---------------|------------|",
        ]
        for asset in all_sessions:
            df = grids.get(asset)
            if df is None:
                lines.append(f"| {asset} | — | — | — | — | — | _no cache_ |")
                continue
            nb = best_neighborhood_params_min_fill_rate(df, MIN_FR)
            if not nb:
                lines.append(f"| {asset} | — | — | — | — | — | _no combo meets 30%_ |")
                continue
            fw_str = f"{int(nb['fill_window'])}s" if nb.get("fill_window") is not None else "—"
            lines.append(
                f"| {asset} | {_price(nb.get('buy'))} | {_price(nb.get('sell'))} | {fw_str} "
                f"| {_edge(nb.get('neighborhood_mean_edge'))} | {_pct(nb.get('pct_positive'))} "
                f"| {_agree_label(nb.get('peak_vs_neighborhood', '—'))} |"
            )

    # ── 3. Directional + prior resolution at ≥30% ───────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Directional Strategy & Prior Resolution at ≥30% Fill Rate",
        "",
        "> Same analysis as the prior resolution report but only strategy combinations",
        "> that achieve ≥30% fill rate are considered.",
        "",
    ]
    lines += _prev_resolution_section(
        all_sessions, buy_range, sell_range, step,
        fill_window_range=fill_window_range, fill_window_step=fill_window_step,
        min_fill_rate=MIN_FR,
    )

    return lines


def _shape_label(shape: str) -> str:
    return {"plateau": "plateau", "moderate": "moderate", "spike": "spike"}.get(shape, "unknown")

def _agree_label(agree: str) -> str:
    return agree if agree else "—"


def write_report(
    per_asset_best: dict[str, dict],
    asset_ranking: pd.DataFrame,
    per_asset_robustness: dict[str, dict] | None = None,
    per_asset_best_nb: dict[str, dict] | None = None,
    per_asset_best_30pct: dict[str, dict] | None = None,
    current_buy: float | None = config.BUY_PRICE,
    current_sell: float | None = config.SELL_PRICE,
    data_source: str = "api",
    capital: float = 500.0,
    position_size_pct: float = config.POSITION_SIZE_PCT,
    spread_cost: float = 0.002,
    all_sessions: dict | None = None,
    fill_windows: list[int] | None = None,
    buy_range: tuple[float, float] = (0.10, 0.49),
    sell_range: tuple[float, float] = (0.45, 0.96),
    step: float = 0.03,
    fill_window_range: tuple[int, int] = (10, 60),
    fill_window_step: int = 10,
) -> list[str]:
    """Write 7 separate markdown research report files. Returns list of paths written."""
    _ensure_reports_dir()

    # Delete any stale report_*.md files from previous runs
    for _old in glob.glob(os.path.join(config.REPORTS_DIR, "report_*.md")):
        os.remove(_old)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    _robustness = per_asset_robustness or {}
    _best_nb    = per_asset_best_nb    or {}
    _best_30    = per_asset_best_30pct or {}

    SESSIONS_PER_DAY = 288  # 24h × 60min / 5min
    position_usdc = capital * position_size_pct

    # ── Recommendation: prefer moderate/plateau over raw edge ─────────────────
    def _asset_priority(a):
        shape = _robustness.get(a, {}).get("shape", "spike")
        edge  = per_asset_best[a].get("edge_per_session", -999)
        rank  = {"plateau": 2, "moderate": 1, "spike": 0, "unknown": 0}
        return (rank.get(shape, 0), edge)

    best_asset = max(per_asset_best, key=_asset_priority) if per_asset_best else None

    written: list[str] = []

    def _write(filename: str, section_name: str, body_lines: list[str]) -> str:
        header = [
            f"# Skeptic Research Report — {section_name}",
            f"_Generated: {now}_",
            "",
        ]
        content = "\n".join(header + body_lines)
        p = os.path.join(config.REPORTS_DIR, filename)
        with open(p, "w") as fh:
            fh.write(content)
        log.info("Research report section written to %s", p)
        return p

    # ── 1. report_summary.md ──────────────────────────────────────────────────
    summary_lines: list[str] = [
        "## TL;DR",
        "",
        "| Asset | Sessions | Fill Rate | Edge/Session | Shape | Note |",
        "|-------|----------|-----------|--------------|-------|------|",
    ]
    for asset, params in per_asset_best.items():
        n_sess = int(params.get("n_sessions", 0))
        fr     = params.get("fill_rate", 0)
        edge   = params.get("edge_per_session")
        shape  = _robustness.get(asset, {}).get("shape", "unknown")
        summary_lines.append(
            f"| {asset} | {n_sess} | {_pct(fr)} | {_edge(edge)} | {_shape_label(shape)} |"
        )

    fw_desc = f"first {config.MONITOR_SECS} seconds"
    if data_source == "prices":
        methodology_lines = [
            "Per-second mid-prices were collected via `collect_prices.py` and stored as CSV files in `data/prices/`.",
            "Each 5-minute window's price series is used to simulate fills and exits.",
            "",
            "### Strategy simulation",
            f"- **Fill** occurs if the mid-price for either UP or DOWN ≤ `buy_threshold` within the {fw_desc}",
            "- **Sell hit** occurs if the mid-price subsequently reaches ≥ `sell_threshold` before window end",
            "- Otherwise position goes to resolution (1.0 or 0.0)",
            "- **Edge per session** = fill_rate × expected_PnL_per_fill",
        ]
    else:
        methodology_lines = [
            "Historical 5-minute UP/DOWN markets were fetched from the Polymarket Gamma API.",
            "Trade data from the CLOB was used to reconstruct intra-session prices.",
            "",
            "> **Note:** Price reconstruction from trades is an approximation.",
            "> Results should be treated as directional signals, not precise predictions.",
            "",
            "### Strategy simulation",
            f"- **Fill** occurs if the minimum trade price for either UP or DOWN ≤ `buy_threshold` within the {fw_desc}",
            "- **Sell hit** occurs if the max subsequent trade price ≥ `sell_threshold`",
            "- Otherwise position goes to resolution (1.0 or 0.0)",
            "- **Edge per session** = fill_rate × expected_PnL_per_fill",
        ]

    summary_lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        *methodology_lines,
    ]

    if current_buy is not None and current_sell is not None and not asset_ranking.empty:
        summary_lines += [
            "",
            "---",
            "",
            f"## Asset Ranking at Current Thresholds (buy={current_buy:.2f}, sell={current_sell:.2f})",
            "",
            asset_ranking.to_markdown(index=False),
        ]

    written.append(_write("report_summary.md", "Summary", summary_lines))

    # ── 2. report_params.md ───────────────────────────────────────────────────
    params_lines: list[str] = [
        "## Optimal Parameters per Asset",
        "",
        "> Parameters with the highest edge across the full grid.",
        "> **Peak vs NB** compares the peak to the best surrounding region — diverge means the peak may be a lucky outlier.",
        "> **Shape** measures whether the peak sits on a broad plateau or is a lone spike.",
        "> **NB Ratio** = neighbor mean edge / peak edge. How much of the peak's edge survives in the surrounding grid points (±1 step in each direction).",
        "> - 1.0 = neighbors match the peak exactly — pure plateau",
        "> - 0.6 = neighbors average 60% of peak edge — reasonably flat",
        "> - 0.0 or negative = neighbors have zero or negative edge — lone spike surrounded by losing combinations, strong overfitting signal",
        "",
        "| Asset | Sessions | Buy | Sell | Fill Win | Fill Rate | Sell Hit% | Edge/Session | Shape | NB Ratio | Peak vs NB |",
        "|-------|----------|-----|------|----------|-----------|-----------|--------------|-------|----------|------------|",
    ]
    for asset, params in per_asset_best.items():
        n_sess = int(params.get("n_sessions", 0))
        buy    = params.get("buy")
        sell   = params.get("sell")
        fw     = f"{int(params['fill_window'])}s" if "fill_window" in params else "60s"
        fr     = params.get("fill_rate")
        shr    = params.get("sell_hit_rate")
        edge   = params.get("edge_per_session")
        rob    = _robustness.get(asset, {})
        shape  = _shape_label(rob.get("shape", "unknown"))
        ratio  = _ratio(rob.get("robustness_ratio"))
        nb     = _best_nb.get(asset, {})
        agree  = _agree_label(nb.get("peak_vs_neighborhood", "—"))
        params_lines.append(
            f"| {asset} | {n_sess} | {_price(buy)} | {_price(sell)} | {fw} "
            f"| {_pct(fr)} | {_pct(shr)} | {_edge(edge)} | {shape} | {ratio} | {agree} |"
        )

    params_lines += [
        "",
        "---",
        "",
        "## Best Neighborhood per Asset",
        "",
        "> The region whose *average* surrounding edge is highest — more trustworthy than the raw peak when they disagree.",
        "> Use these parameters instead of the peak when **Peak vs NB** shows diverge.",
        "",
        "| Asset | NB Buy | NB Sell | NB Fill Win | NB Mean Edge | % NB Positive | Peak Edge | Peak vs NB |",
        "|-------|--------|---------|-------------|--------------|---------------|-----------|------------|",
    ]
    for asset in per_asset_best:
        nb = _best_nb.get(asset, {})
        if not nb:
            params_lines.append(f"| {asset} | — | — | — | — | — | — | — |")
            continue
        nb_fw = f"{nb['fill_window']}s" if nb.get("fill_window") is not None else "—"
        pct_p = _pct(nb.get("pct_positive"))
        agree = _agree_label(nb.get("peak_vs_neighborhood", "—"))
        params_lines.append(
            f"| {asset} | {_price(nb.get('buy'))} | {_price(nb.get('sell'))} | {nb_fw} "
            f"| {_edge(nb.get('neighborhood_mean_edge'))} | {pct_p} "
            f"| {_edge(nb.get('peak_edge'))} | {agree} |"
        )

    params_lines += [
        "",
        "---",
        "",
        "## Optimal Parameters at ≥30% Fill Rate",
        "",
        "> Only combinations where fill_rate ≥ 30% are eligible.",
        "> Low fill rates can produce high edge from a small lucky sample.",
        "> This table forces the strategy to trigger regularly.",
        "",
        "| Asset | Buy | Sell | Fill Win | Fill Rate | Sell Hit% | Edge/Session |",
        "|-------|-----|------|----------|-----------|-----------|--------------|",
    ]
    for asset in per_asset_best:
        p30 = _best_30.get(asset, {})
        if not p30:
            params_lines.append(f"| {asset} | — | — | — | — | — | _no combo meets 30% fill rate_ |")
            continue
        fw30 = f"{int(p30['fill_window'])}s" if "fill_window" in p30 else "60s"
        params_lines.append(
            f"| {asset} | {_price(p30.get('buy'))} | {_price(p30.get('sell'))} | {fw30} "
            f"| {_pct(p30.get('fill_rate'))} | {_pct(p30.get('sell_hit_rate'))} "
            f"| {_edge(p30.get('edge_per_session'))} |"
        )

    written.append(_write("report_params.md", "Parameters", params_lines))

    # ── 3. report_profit.md ───────────────────────────────────────────────────
    profit_lines: list[str] = [
        "## Estimated Profit",
        "",
        f"_Assumptions: ${capital:,.0f} starting capital, {position_size_pct:.0%} per trade, {SESSIONS_PER_DAY} windows/day._",
        f"_Spread cost: {spread_cost:.3f}/share/crossing. Polymarket charges 0% maker/taker fees._",
        "",
        "> **These numbers are theoretical upper bounds.** They assume the in-sample edge holds",
        "> perfectly out-of-sample, every session runs, and execution matches simulation exactly.",
        "> Treat them as a relative ranking between assets, not a cash forecast.",
        "> Assets with shape=spike are especially likely to underperform these estimates.",
        "",
        "| Asset | Buy | Sell | Shape | $/Session (net) | $/Day | $/Week | $/Month |",
        "|-------|-----|------|-------|-----------------|-------|--------|---------|",
    ]
    for asset, params in per_asset_best.items():
        buy       = params.get("buy")
        sell      = params.get("sell")
        edge      = params.get("edge_per_session")
        fill_rate = params.get("fill_rate", 0)
        shr       = params.get("sell_hit_rate", 0)
        shape     = _shape_label(_robustness.get(asset, {}).get("shape", "unknown"))
        if buy and sell and edge is not None and buy > 0:
            shares  = position_usdc / buy
            gross   = edge * shares
            spread_ = fill_rate * spread_cost * shares + fill_rate * shr * spread_cost * shares
            net     = gross - spread_
            profit_lines.append(
                f"| {asset} | {_price(buy)} | {_price(sell)} | {shape} "
                f"| ${net:+.4f} | ${net * SESSIONS_PER_DAY:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 7:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 30:+.2f} |"
            )

    profit_lines += [
        "",
        "### Best Neighborhood Params",
        "",
        "_Uses the neighborhood mean edge (average of surrounding grid points) as the edge estimate._",
        "_Fill rate is approximated from the peak params for each asset._",
        "",
        "| Asset | NB Buy | NB Sell | NB Fill Win | NB Mean Edge | $/Session (net) | $/Day | $/Week | $/Month |",
        "|-------|--------|---------|-------------|-------------|-----------------|-------|--------|---------|",
    ]
    for asset, nb in _best_nb.items():
        nb_buy    = nb.get("buy")
        nb_sell   = nb.get("sell")
        nb_edge   = nb.get("neighborhood_mean_edge")
        nb_fw     = nb.get("fill_window")
        peak      = per_asset_best.get(asset, {})
        fill_rate = peak.get("fill_rate", 0)
        shr       = peak.get("sell_hit_rate", 0)
        fw_str    = f"{int(nb_fw)}s" if nb_fw is not None else "—"
        if nb_buy and nb_sell and nb_edge is not None and nb_buy > 0:
            shares  = position_usdc / nb_buy
            gross   = nb_edge * shares
            spread_ = fill_rate * spread_cost * shares + fill_rate * shr * spread_cost * shares
            net     = gross - spread_
            profit_lines.append(
                f"| {asset} | {_price(nb_buy)} | {_price(nb_sell)} | {fw_str} "
                f"| {_edge(nb_edge)} "
                f"| ${net:+.4f} | ${net * SESSIONS_PER_DAY:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 7:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 30:+.2f} |"
            )

    written.append(_write("report_profit.md", "Profit Estimates", profit_lines))

    # ── 4. report_time_of_day.md — volatility + edge combined ────────────────
    if all_sessions:
        _fws = fill_windows if fill_windows else [20, 40, 60]
        tod_lines: list[str] = []

        vol_lines = _compute_hourly_volatility(all_sessions, _fws, bucket_hours=3)
        if vol_lines:
            tod_lines += [
                "## Volatility by Time of Day (3-Hour Buckets)",
                "",
                "> Mean price range (max − min) within the first N seconds of each 5-minute window,",
                "> grouped into 3-hour buckets. CT = UTC−6 (Central Standard Time).",
                "> Sorted by mean volatility descending.",
                "",
                *vol_lines,
            ]

        if _best_nb:
            edge_lines = _edge_by_hour(all_sessions, _best_nb, bucket_hours=3)
            if edge_lines:
                tod_lines += [
                    "",
                    "---",
                    "",
                    "## Edge by Time of Day — Best Neighborhood Strategy (3-Hour Buckets)",
                    "",
                    "> Edge per session for each asset using its best neighborhood buy/sell/fill_window params.",
                    "> Positive = expected profit per window in that time slot.",
                    "",
                    *edge_lines,
                ]

        if tod_lines:
            written.append(_write("report_time_of_day.md", "Time of Day Analysis", tod_lines))

    # ── 6. report_prior_resolution.md (only if all_sessions present) ──────────
    if all_sessions:
        pr_lines = _prev_resolution_section(
            all_sessions, buy_range, sell_range, step,
            fill_window_range=fill_window_range, fill_window_step=fill_window_step,
        )
        if pr_lines:
            prior_res_lines: list[str] = [
                "## Strategy by Prior Window Resolution",
                "",
                "> Sessions are split by whether the immediately preceding 5-minute window",
                "> resolved UP or DOWN. Only sessions with a confirmed adjacent prior window",
                "> are included — gaps in data are discarded.",
                "> **Bet UP only** = only enter when the UP price dips to buy threshold.",
                "> **Bet DOWN only** = only enter when the DOWN price dips to buy threshold.",
                "> **Both** = original strategy (first side to fill wins).",
                "",
                *pr_lines,
            ]
            written.append(_write("report_prior_resolution.md", "Strategy by Prior Window Resolution", prior_res_lines))

    # ── 7. report_30pct.md (only if all_sessions present) ────────────────────
    if all_sessions and _best_30:
        lines_30 = _report_30pct_section(
            all_sessions, _best_30, buy_range, sell_range, step,
            fill_window_range=fill_window_range, fill_window_step=fill_window_step,
        )
        if lines_30:
            written.append(_write("report_30pct.md", "≥30% Fill Rate Analysis", lines_30))

    # ── 8. report_recommendation.md ───────────────────────────────────────────
    rec_lines: list[str] = [
        "## Recommendation",
        "",
    ]

    if best_asset:
        bp        = per_asset_best[best_asset]
        buy       = bp.get("buy", current_buy)
        sell      = bp.get("sell", current_sell)
        edge      = bp.get("edge_per_session", 0)
        fill_rate = bp.get("fill_rate", 0)
        shr       = bp.get("sell_hit_rate", 0)
        best_fw   = int(bp["fill_window"]) if "fill_window" in bp else 60
        rob       = _robustness.get(best_asset, {})
        shape     = rob.get("shape", "unknown")
        rob_ratio = rob.get("robustness_ratio")
        rob_pct   = rob.get("pct_positive")
        rob_mean  = rob.get("neighbor_mean_edge")
        nb        = _best_nb.get(best_asset, {})

        def _net(b, e, fr, s):
            if not b or e is None:
                return 0.0, 0.0
            sh = position_usdc / b
            sp = fr * spread_cost * sh + fr * s * spread_cost * sh
            ns = e * sh - sp
            return ns, ns * SESSIONS_PER_DAY

        net_session, net_day = _net(buy, edge, fill_rate, shr)

        rob_line = (
            f"{_shape_label(shape)} — ratio={_ratio(rob_ratio)}, "
            f"neighbor mean edge={_edge(rob_mean)}, "
            f"{_pct(rob_pct)} of neighbors positive"
            if rob_ratio is not None and rob_pct is not None
            else "_not available_"
        )

        flags = []
        if shape == "spike":
            flags.append("> Note: shape is spike — the peak has weak neighborhood support. See Best Neighborhood recommendation below.")
        if best_fw < 30:
            flags.append(f"> Note: fill window is {best_fw}s — very short; verify this isn't a noise artifact.")

        rec_lines += [
            f"**Best asset:** {best_asset}",
            "",
            "### Option A — Peak Threshold (highest raw edge)",
            "",
            f"| | Value |",
            f"|---|---|",
            f"| Buy | {_price(buy)} |",
            f"| Sell | {_price(sell)} |",
            f"| Fill window | {best_fw}s |",
            f"| Edge/session | {_edge(edge)} |",
            f"| Fill rate | {_pct(fill_rate)} |",
            f"| Sell hit rate | {_pct(shr)} |",
            f"| Est. $/session (net) | ${net_session:+.4f} |",
            f"| Est. $/day | ${net_day:+.2f} |",
            f"| Robustness | {rob_line} |",
            "",
        ]
        if flags:
            rec_lines += flags
            rec_lines.append("")

        rec_lines += [
            "```python",
            f"# Option A — config.py",
            f"BUY_PRICE    = {buy}",
            f"SELL_PRICE   = {sell}",
            f"MONITOR_SECS = {best_fw}",
            "```",
            "",
        ]

        if nb:
            nb_buy  = nb.get("buy")
            nb_sell = nb.get("sell")
            nb_fw   = nb.get("fill_window", best_fw)
            nb_edge = nb.get("neighborhood_mean_edge")
            nb_pct  = nb.get("pct_positive")
            nb_ns, nb_nd = _net(nb_buy, nb_edge, fill_rate, shr)
            nb_agree = nb.get("peak_vs_neighborhood", "—")

            rec_lines += [
                "### Option B — Best Neighborhood (most robust region)",
                "",
                "> Uses the center of the region with the highest average surrounding edge.",
                "> Prefer this when shape=spike or peak vs NB=diverge.",
                "",
                f"| | Value |",
                f"|---|---|",
                f"| Buy | {_price(nb_buy)} |",
                f"| Sell | {_price(nb_sell)} |",
                f"| Fill window | {nb_fw}s |",
                f"| NB mean edge | {_edge(nb_edge)} |",
                f"| % NB positive | {_pct(nb_pct)} |",
                f"| Est. $/session (net, approx) | ${nb_ns:+.4f} |",
                f"| Est. $/day (approx) | ${nb_nd:+.2f} |",
                f"| Peak vs NB | {nb_agree} |",
                "",
                "```python",
                f"# Option B — config.py",
                f"BUY_PRICE    = {nb_buy}",
                f"SELL_PRICE   = {nb_sell}",
                f"MONITOR_SECS = {nb_fw}",
                "```",
            ]
    else:
        rec_lines.append("_Insufficient data to make a recommendation._")

    written.append(_write("report_recommendation.md", "Recommendation", rec_lines))

    return written
