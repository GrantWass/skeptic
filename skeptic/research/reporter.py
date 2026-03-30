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
                                        neighborhood_robustness, best_neighborhood_params_min_fill_rate,
                                        sweep_high_buy, grid_search_high_buy, high_buy_hurst,
                                        analyze_timing_buckets)

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

    if all_sessions:
        written.append(write_high_buy_report(all_sessions))

    return written


def _high_buy_section(all_sessions: dict) -> list[str]:
    """
    Sweep buy thresholds (0.65–0.95) for each asset, combining UP and DOWN fills.
    Shows both anytime and last-2-min-only modes side by side.
    5-cent slippage is baked in to all payoff calculations.
    """
    LAST2_SECS   = 180
    EXCL_LAST_30 = 270  # exclude final 30 seconds (max_elapsed_secs=270)
    SLIPPAGE     = 0.05

    def _nb_mean_edge(df: "pd.DataFrame", t: float) -> "float | None":
        neighbors = df[df["threshold"].between(round(t - 0.05, 2), round(t + 0.05, 2))]
        return float(neighbors["edge_per_session"].mean()) if not neighbors.empty else None

    def _sweep_with_nb(sessions: list, min_elapsed: int = 0, max_elapsed: int = 300) -> "pd.DataFrame":
        df = sweep_high_buy(sessions, min_elapsed_secs=min_elapsed, max_elapsed_secs=max_elapsed, slippage=SLIPPAGE)
        df["nb_mean_edge"] = df["threshold"].apply(lambda t: _nb_mean_edge(df, t))
        return df

    def _best_row(df: "pd.DataFrame") -> "pd.Series | None":
        nb_sorted = df.dropna(subset=["nb_mean_edge"]).sort_values("nb_mean_edge", ascending=False)
        return nb_sorted.iloc[0] if not nb_sorted.empty and nb_sorted.iloc[0]["n_fills"] > 0 else None

    lines: list[str] = [
        "## High-Probability Buy Sweep",
        "",
        "> Buys either UP or DOWN whenever that token's price touches the threshold (first trigger per session).",
        "> Sweeps thresholds 0.65–0.95 in steps of 0.05.",
        f"> **{int(SLIPPAGE * 100)}-cent slippage assumed**: effective fill price = T + {SLIPPAGE:.2f}.",
        "> Payoff at effective price E: **+(1−E)** if that direction resolves, **−E** otherwise.",
        "> Break-even win rate equals E (e.g. threshold=0.80 fills at 0.83, requires >83% wins).",
        "> **NB Mean Edge** = average edge across T−0.05, T, T+0.05 — more robust than the raw peak.",
        "> **Last 2 min** = only fills that trigger in the final 2 minutes of the window (≥180s elapsed).",
        "> **−30s** = same as above but excluding triggers in the final 30 seconds (avoids thin-book spikes).",
        "",
        "### Best Threshold per Asset (by NB Mean Edge)",
        "",
        "| Asset | Mode | Best Buy | Fills | Fill Rate | Win Rate | Edge/Session | NB Mean Edge |",
        "|-------|------|----------|-------|-----------|----------|--------------|--------------|",
    ]

    per_asset_any: dict[str, "pd.DataFrame"] = {}
    per_asset_l2m: dict[str, "pd.DataFrame"] = {}
    per_asset_any30: dict[str, "pd.DataFrame"] = {}
    per_asset_l2m30: dict[str, "pd.DataFrame"] = {}

    for asset, sessions in all_sessions.items():
        if not sessions:
            continue
        df_any    = _sweep_with_nb(sessions, min_elapsed=0)
        df_l2m    = _sweep_with_nb(sessions, min_elapsed=LAST2_SECS)
        df_any30  = _sweep_with_nb(sessions, min_elapsed=0,          max_elapsed=EXCL_LAST_30)
        df_l2m30  = _sweep_with_nb(sessions, min_elapsed=LAST2_SECS, max_elapsed=EXCL_LAST_30)
        per_asset_any[asset]   = df_any
        per_asset_l2m[asset]   = df_l2m
        per_asset_any30[asset] = df_any30
        per_asset_l2m30[asset] = df_l2m30

        for label, df in [("Any time", df_any), ("Last 2 min", df_l2m),
                          ("Any time −30s", df_any30), ("Last 2 min −30s", df_l2m30)]:
            best = _best_row(df)
            if best is None:
                lines.append(f"| {asset} | {label} | — | — | — | — | — | — |")
                continue
            t = best["threshold"]
            win_str = f"{best['win_rate']:.1%}" if best["win_rate"] is not None else "—"
            nb_str  = f"{best['nb_mean_edge']:+.4f}" if pd.notna(best["nb_mean_edge"]) else "—"
            lines.append(
                f"| {asset} | {label} | **{t:.2f}** | {int(best['n_fills'])} "
                f"| {best['fill_rate']:.1%} | {win_str} "
                f"| {best['edge_per_session']:+.4f} | {nb_str} |"
            )

    lines += [""]

    # ── Grid search: threshold × window cutoff ────────────────────────────────
    lines += [
        "---",
        "",
        "## Grid Search: Threshold × Entry Window (Edge/Session)",
        "",
        "> Rows = how far into the window before entering. Columns = buy threshold.",
        "> Values = edge per session. Blank = no fills at that combination.",
        "",
    ]

    cutoffs = list(range(0, 271, 30))
    thresholds_grid = [round(t / 100, 2) for t in range(65, 97, 5)]

    def _cutoff_label(c: int) -> str:
        if c == 0:
            return "Any time"
        remaining = 300 - c
        m, s = divmod(remaining, 60)
        return f"Last {m}m" if s == 0 else f"Last {m}m{s}s"

    for asset, sessions in all_sessions.items():
        if not sessions:
            continue
        lines += [f"### {asset}", ""]
        df_grid = grid_search_high_buy(sessions, thresholds=thresholds_grid, cutoffs=cutoffs, slippage=SLIPPAGE)
        pivot = df_grid.pivot(index="cutoff_secs", columns="threshold", values="edge_per_session")

        header = "| Entry point | " + " | ".join(f"T={t:.2f}" for t in thresholds_grid) + " |"
        sep    = "|-------------|" + "|".join("--------" for _ in thresholds_grid) + "|"
        lines += [header, sep]

        for c in cutoffs:
            row_label = _cutoff_label(c)
            cells = []
            for t in thresholds_grid:
                val = pivot.loc[c, t] if c in pivot.index and t in pivot.columns else None
                n_fills = df_grid[(df_grid["cutoff_secs"] == c) & (df_grid["threshold"] == t)]["n_fills"].values
                if val is None or (len(n_fills) > 0 and n_fills[0] == 0):
                    cells.append("—")
                else:
                    cells.append(f"{val:+.4f}")
            lines.append(f"| {row_label} | " + " | ".join(cells) + " |")

        lines += [""]

    # ── Hurst exponent section ────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Hurst Exponent Analysis",
        "",
        "> Measures whether price series within each 5-minute window are **trending** or **mean-reverting**.",
        "> Computed on the combined UP+DOWN price series per session.",
        "> **H > 0.5** → trending (momentum): prices that spike tend to stay high → good for high-buy.",
        "> **H ≈ 0.5** → random walk: no memory.",
        "> **H < 0.5** → mean-reverting: spikes tend to revert → bad for high-buy.",
        "> **Filled H** = mean H for sessions where price touched the threshold.",
        "> If Filled H > All H, the strategy is selectively entering trending sessions.",
        "",
        "### Any Time",
        "",
        "| Asset | All H | Filled H | Unfilled H | Filled − All | Interpretation |",
        "|-------|-------|----------|------------|--------------|----------------|",
    ]

    def _hurst_interp(filled_h: "float | None", all_h: "float | None") -> str:
        if filled_h is None or all_h is None:
            return "—"
        diff = filled_h - all_h
        if filled_h > 0.55:
            trend = "trending fills"
        elif filled_h < 0.45:
            trend = "mean-reverting fills ⚠️"
        else:
            trend = "random-walk fills"
        if diff > 0.02:
            return f"{trend} (selects high-momentum)"
        elif diff < -0.02:
            return f"{trend} (selects low-momentum)"
        return f"{trend} (neutral selection)"

    best_thresholds: dict[str, float] = {}
    for asset, df in per_asset_any.items():
        best = _best_row(df)
        best_thresholds[asset] = best["threshold"] if best is not None else 0.80

    for asset, sessions in all_sessions.items():
        if not sessions:
            continue
        t = best_thresholds.get(asset, 0.80)
        h = high_buy_hurst(sessions, threshold=t, min_elapsed_secs=0)
        diff_str = f"{(h['filled_h'] or 0) - (h['all_h'] or 0):+.4f}" if h["filled_h"] and h["all_h"] else "—"
        interp = _hurst_interp(h["filled_h"], h["all_h"])
        lines.append(
            f"| {asset} (T={t:.2f}) "
            f"| {h['all_h'] or '—'} "
            f"| {h['filled_h'] or '—'} "
            f"| {h['unfilled_h'] or '—'} "
            f"| {diff_str} "
            f"| {interp} |"
        )

    lines += [
        "",
        "### Last 2 Min",
        "",
        "| Asset | All H | Filled H | Unfilled H | Filled − All | Interpretation |",
        "|-------|-------|----------|------------|--------------|----------------|",
    ]

    for asset, sessions in all_sessions.items():
        if not sessions:
            continue
        t = best_thresholds.get(asset, 0.80)
        h = high_buy_hurst(sessions, threshold=t, min_elapsed_secs=LAST2_SECS)
        diff_str = f"{(h['filled_h'] or 0) - (h['all_h'] or 0):+.4f}" if h["filled_h"] and h["all_h"] else "—"
        interp = _hurst_interp(h["filled_h"], h["all_h"])
        lines.append(
            f"| {asset} (T={t:.2f}) "
            f"| {h['all_h'] or '—'} "
            f"| {h['filled_h'] or '—'} "
            f"| {h['unfilled_h'] or '—'} "
            f"| {diff_str} "
            f"| {interp} |"
        )

    lines += [""]
    return lines


def write_high_buy_report(all_sessions: dict) -> str:
    """Generate only report_high_buy.md. Skips all grid searches."""
    _ensure_reports_dir()
    lines = _high_buy_section(all_sessions)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    header = ["# Skeptic Research Report — High-Probability Buy Sweep", f"_Generated: {now}_", ""]
    content = "\n".join(header + lines)
    path = os.path.join(config.REPORTS_DIR, "report_high_buy.md")
    with open(path, "w") as fh:
        fh.write(content)
    log.info("Research report written to %s", path)
    return path


def _trigger_timing_section(all_sessions: dict) -> list[str]:
    """
    For key thresholds, show win rate and edge per fill bucketed by when the
    trigger actually fires within the 5-minute window, plus cross-asset analysis.
    """
    THRESHOLDS  = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    SLIPPAGE    = 0.05
    BUCKET_SECS = 60
    MIN_FILLS   = 10  # minimum fills for a cell to count in analysis

    bucket_labels = [f"{i*BUCKET_SECS}–{(i+1)*BUCKET_SECS}s" for i in range(5)]
    header = "| Threshold | " + " | ".join(bucket_labels) + " | Best Window |"
    sep    = "|-----------|" + "|".join(["----------"] * 5) + "|-------------|"

    lines: list[str] = [
        "## Trigger Timing Analysis",
        "",
        "> Win rate and fill count when the trigger fires in each 60-second bucket of the window.",
        "> Format: `win% (N fills)`. **Best Window** = bucket with highest win rate (min 10 fills).",
        f"> {int(SLIPPAGE * 100)}-cent slippage baked in to edge calculations.",
        "",
    ]

    # Collect structured data for the analysis section
    # records[asset][threshold][bucket_label] = {win_rate, edge, n_fills}
    records: dict = {}

    for asset, sessions in all_sessions.items():
        if not sessions:
            continue
        total_sessions = len(sessions)
        records[asset] = {}
        lines += [f"### {asset} ({total_sessions} sessions)", "", header, sep]

        for t in THRESHOLDS:
            df = analyze_timing_buckets(sessions, threshold=t, bucket_secs=BUCKET_SECS, slippage=SLIPPAGE)
            records[asset][t] = {}
            cells: list[str] = []
            best_win: float = -1.0
            best_label: str = "—"
            for _, row in df.iterrows():
                n = int(row["n_fills"])
                bl = str(row["bucket_label"])
                epf = row["edge_per_fill"]
                eps = (epf * n / total_sessions) if (epf is not None and total_sessions > 0) else None
                records[asset][t][bl] = {
                    "n_fills":        n,
                    "win_rate":       row["win_rate"],
                    "edge_per_fill":  epf,
                    "edge_per_session": eps,
                }
                if n == 0:
                    cells.append("—")
                else:
                    wr = row["win_rate"]
                    cells.append(f"{wr:.1%} ({n})")
                    if wr is not None and n >= MIN_FILLS and wr > best_win:
                        best_win  = wr
                        best_label = bl
            lines.append("| " + f"{t:.2f}" + " | " + " | ".join(cells) + f" | {best_label} |")

        # Edge-per-fill and edge-per-session tables
        lines += ["", "_Edge/fill  |  Edge/session at each bucket (edge/session = edge/fill × fill_rate):_", ""]
        edge_header = "| Threshold | " + " | ".join(bucket_labels) + " |"
        edge_sep    = "|-----------|" + "|".join(["---------- "] * 5) + "|"
        lines += ["_Edge/fill:_", "", edge_header, edge_sep]
        for t in THRESHOLDS:
            cells = []
            for bl in bucket_labels:
                e = records[asset][t].get(bl, {}).get("edge_per_fill")
                n = records[asset][t].get(bl, {}).get("n_fills", 0)
                cells.append(f"{e:+.4f}" if (e is not None and n >= MIN_FILLS) else "—")
            lines.append("| " + f"{t:.2f}" + " | " + " | ".join(cells) + " |")
        lines += ["", "_Edge/session:_", "", edge_header, edge_sep]
        for t in THRESHOLDS:
            cells = []
            for bl in bucket_labels:
                eps = records[asset][t].get(bl, {}).get("edge_per_session")
                n   = records[asset][t].get(bl, {}).get("n_fills", 0)
                cells.append(f"{eps:+.5f}" if (eps is not None and n >= MIN_FILLS) else "—")
            lines.append("| " + f"{t:.2f}" + " | " + " | ".join(cells) + " |")
        lines += [""]

    # ── Cross-asset analysis ──────────────────────────────────────────────────
    if not records:
        return lines

    # ── Helpers: neighbour-smoothed edge (fill and session) ──────────────────
    # Averages the bucket with its immediate neighbours to reduce overfitting.
    def _nb_edge(thresh_data: dict, t: float, idx: int) -> float | None:
        """Smoothed edge/fill — quality per trade."""
        indices = [i for i in (idx - 1, idx, idx + 1) if 0 <= i < len(bucket_labels)]
        vals = []
        for i in indices:
            cell = thresh_data.get(t, {}).get(bucket_labels[i], {})
            if cell.get("n_fills", 0) >= MIN_FILLS and cell.get("edge_per_fill") is not None:
                vals.append(cell["edge_per_fill"])
        return sum(vals) / len(vals) if vals else None

    def _nb_edge_session(thresh_data: dict, t: float, idx: int) -> float | None:
        """Smoothed edge/session — actual P&L impact per window (fill quality × fill rate)."""
        indices = [i for i in (idx - 1, idx, idx + 1) if 0 <= i < len(bucket_labels)]
        vals = []
        for i in indices:
            cell = thresh_data.get(t, {}).get(bucket_labels[i], {})
            if cell.get("n_fills", 0) >= MIN_FILLS and cell.get("edge_per_session") is not None:
                vals.append(cell["edge_per_session"])
        return sum(vals) / len(vals) if vals else None

    lines += [
        "---",
        "",
        "## Analysis",
        "",
        "> All edge figures use **neighbour-smoothed** values: each bucket is averaged with its",
        "> immediate neighbours (±60 s) to reduce single-bucket overfitting.",
        "",
    ]

    # ── 1. Best threshold per bucket, per asset ───────────────────────────────
    lines += [
        "### Best Threshold per Bucket (neighbour-smoothed, ranked by Edge/Session)",
        "",
        "> For each 60-second window bucket, which threshold produced the best smoothed edge/session?",
        "> Edge/session = edge/fill × fill_rate. A bucket with great edge/fill but 0.5% fill rate",
        "> barely moves your P&L — edge/session captures actual impact per window.",
        "",
    ]

    for asset, thresh_data in records.items():
        lines += [f"#### {asset}", ""]
        tbl_header = "| Bucket | Best T | Sm. Edge/Session | Sm. Edge/Fill | Win Rate | Fills |"
        tbl_sep    = "|--------|--------|-----------------|--------------|----------|-------|"
        lines += [tbl_header, tbl_sep]

        for idx, bl in enumerate(bucket_labels):
            best_t_nb: float | None = None
            best_nb_eps: float = float("-inf")
            for t in THRESHOLDS:
                nb_eps = _nb_edge_session(thresh_data, t, idx)
                if nb_eps is not None and nb_eps > best_nb_eps:
                    best_nb_eps = nb_eps
                    best_t_nb = t

            if best_t_nb is None:
                lines.append(f"| {bl} | — | — | — | — | — |")
            else:
                cell  = thresh_data.get(best_t_nb, {}).get(bl, {})
                nb_epf = _nb_edge(thresh_data, best_t_nb, idx)
                wr    = cell.get("win_rate")
                n     = cell.get("n_fills", 0)
                epf_str = f"{nb_epf:+.4f}" if nb_epf is not None else "—"
                wr_str  = f"{wr:.1%}" if wr is not None and n >= MIN_FILLS else "—"
                lines.append(
                    f"| {bl} | **T={best_t_nb:.2f}** | {best_nb_eps:+.5f} "
                    f"| {epf_str} | {wr_str} | {n} |"
                )
        lines += [""]

    # ── 2. Buckets to avoid per asset ─────────────────────────────────────────
    lines += [
        "### Buckets to Avoid per Asset",
        "",
        "> A bucket is flagged as **avoid** when the best smoothed edge/session across all",
        "> thresholds is still ≤ 0 (no threshold is reliably profitable in that window).",
        "",
    ]

    for asset, thresh_data in records.items():
        avoid_buckets: list[str] = []
        trade_buckets: list[tuple[str, float, float, float]] = []  # (bucket, best_t, sm_eps, sm_epf)

        for idx, bl in enumerate(bucket_labels):
            best_nb_eps = float("-inf")
            best_t_here: float | None = None
            for t in THRESHOLDS:
                nb_eps = _nb_edge_session(thresh_data, t, idx)
                if nb_eps is not None and nb_eps > best_nb_eps:
                    best_nb_eps = nb_eps
                    best_t_here = t

            if best_t_here is None or best_nb_eps <= 0:
                avoid_buckets.append(bl)
            else:
                nb_epf = _nb_edge(thresh_data, best_t_here, idx) or 0.0
                trade_buckets.append((bl, best_t_here, best_nb_eps, nb_epf))

        if avoid_buckets:
            lines.append(f"**{asset}** — avoid: {', '.join(f'`{b}`' for b in avoid_buckets)}")
        else:
            lines.append(f"**{asset}** — all buckets show positive edge/session (no avoid zones)")

        if trade_buckets:
            trade_strs = [
                f"`{bl}` T={t:.2f} (eps={eps:+.5f}, epf={epf:+.4f})"
                for bl, t, eps, epf in trade_buckets
            ]
            lines.append(f"  Trade zones: {', '.join(trade_strs)}")
        lines += [""]

    # ── 3. Actionable trading recommendation per asset ───────────────────────
    lines += [
        "### Trading Recommendation per Asset",
        "",
        "> Ranked by **edge/session** (actual P&L per window). Edge/fill shown for context.",
        "> ⚠️ sensitivity warning fires when the spread between best/worst threshold edge/session > 0.0005.",
        "",
    ]

    for asset, thresh_data in records.items():
        lines += [f"#### {asset}", ""]

        steps: list[str] = []
        for idx, bl in enumerate(bucket_labels):
            # Rank thresholds by smoothed edge/session
            bucket_nb: list[tuple[float, float, float]] = []  # (threshold, sm_eps, sm_epf)
            for t in THRESHOLDS:
                nb_eps = _nb_edge_session(thresh_data, t, idx)
                nb_epf = _nb_edge(thresh_data, t, idx)
                if nb_eps is not None:
                    bucket_nb.append((t, nb_eps, nb_epf or 0.0))

            if not bucket_nb:
                steps.append(f"- **{bl}** — Skip (no fills)")
                continue

            bucket_nb.sort(key=lambda x: x[1], reverse=True)
            best_t_b, best_nb_eps, best_nb_epf = bucket_nb[0]
            worst_nb_eps = bucket_nb[-1][1]

            if best_nb_eps <= 0:
                steps.append(
                    f"- **{bl}** — ⛔ Skip  "
                    f"(best edge/session still {best_nb_eps:+.5f} at T={best_t_b:.2f})"
                )
            else:
                cell = thresh_data.get(best_t_b, {}).get(bl, {})
                wr   = cell.get("win_rate")
                n    = cell.get("n_fills", 0)
                be   = best_t_b + SLIPPAGE
                wr_str = f"{wr:.1%}" if wr is not None and n >= MIN_FILLS else "?"

                spread = best_nb_eps - worst_nb_eps
                note = ""
                if spread > 0.0005:
                    note = "  ⚠️ High threshold sensitivity"

                steps.append(
                    f"- **{bl}** — ✅ Use T={best_t_b:.2f}  "
                    f"(edge/session {best_nb_eps:+.5f}, edge/fill {best_nb_epf:+.4f}, "
                    f"win {wr_str} vs {be:.0%} B/E, {n} fills){note}"
                )

                # Alt: lower threshold with similar edge/session but more fills
                for t2, eps2, epf2 in bucket_nb[1:]:
                    if eps2 > 0 and eps2 >= best_nb_eps * 0.85:
                        cell2 = thresh_data.get(t2, {}).get(bl, {})
                        n2 = cell2.get("n_fills", 0)
                        if n2 > n * 1.5:
                            steps.append(
                                f"  _Alt: T={t2:.2f} — edge/session {eps2:+.5f}, "
                                f"edge/fill {epf2:+.4f}, {n2} fills (better fill rate)_"
                            )
                            break

        lines += steps + [""]

    return lines


def write_trigger_timing_report(all_sessions: dict) -> str:
    """Generate report_trigger_timing.md — win rate by elapsed seconds at trigger."""
    _ensure_reports_dir()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    header = ["# Skeptic Research Report — Trigger Timing Analysis", f"_Generated: {now}_", ""]
    content = "\n".join(header + _trigger_timing_section(all_sessions))
    path = os.path.join(config.REPORTS_DIR, "report_trigger_timing.md")
    with open(path, "w") as fh:
        fh.write(content)
    log.info("Trigger timing report written to %s", path)
    return path
