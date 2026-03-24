"""
Remove trading windows with fewer than MIN_POINTS data points from price CSV files.

Usage:
    python scripts/filter_incomplete_windows.py [--min-points 280] [--prices-dir data/prices] [--dry-run]
"""
import argparse
import csv
import io
import os
from collections import defaultdict
from pathlib import Path


MIN_POINTS_DEFAULT = 280


def filter_csv(path: Path, min_points: int, dry_run: bool) -> tuple[int, int]:
    """
    Filter rows from a single CSV file, removing any (asset, window_ts) groups
    with fewer than min_points rows.

    Returns (windows_removed, rows_removed).
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Count rows per (asset, window_ts)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (row.get("asset", ""), row.get("window_ts", ""))
        counts[key] += 1

    small_windows = {k for k, n in counts.items() if n < min_points}
    total_windows = len(counts)
    kept_windows = total_windows - len(small_windows)

    kept = [r for r in rows if (r.get("asset", ""), r.get("window_ts", "")) not in small_windows]
    rows_removed = len(rows) - len(kept)
    windows_removed = len(small_windows)

    if not dry_run and small_windows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

    return windows_removed, rows_removed, kept_windows, len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter incomplete trading windows from price CSVs.")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS_DEFAULT,
                        help=f"Minimum data points per window (default: {MIN_POINTS_DEFAULT})")
    parser.add_argument("--prices-dir", default="data/prices",
                        help="Directory containing price CSV files (default: data/prices)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without modifying files")
    args = parser.parse_args()

    prices_path = Path(args.prices_dir)
    if not prices_path.exists():
        print(f"Error: prices directory not found: {prices_path}")
        return

    csv_files = sorted(prices_path.glob("prices_*.csv"))
    if not csv_files:
        print(f"No price CSV files found in {prices_path}")
        return

    total_windows_removed = 0
    total_rows_removed = 0
    total_windows_kept = 0
    total_rows_kept = 0
    for csv_file in csv_files:
        windows_removed, rows_removed, windows_kept, rows_kept = filter_csv(csv_file, args.min_points, args.dry_run)
        verb = "remove" if args.dry_run else "removed"
        print(
            f"{csv_file.name}: "
            f"would {verb} {windows_removed} window(s) ({rows_removed} rows)  |  "
            f"keep {windows_kept} window(s) ({rows_kept} rows)"
        )
        total_windows_removed += windows_removed
        total_rows_removed += rows_removed
        total_windows_kept += windows_kept
        total_rows_kept += rows_kept

    print(f"\n{'Dry run' if args.dry_run else 'Done'} — "
          f"remove {total_windows_removed} window(s) ({total_rows_removed} rows)  |  "
          f"keep {total_windows_kept} window(s) ({total_rows_kept} rows)")


if __name__ == "__main__":
    main()
