"""
Remove trading windows with fewer than MIN_POINTS data points from price CSV files.

Usage:
    python scripts/filter_incomplete_windows.py [--min-points 280] [--prices-dir data/prices] [--dry-run]
"""
import argparse

from skeptic import storage


MIN_POINTS_DEFAULT = 280


def filter_csv(path: str, min_points: int, dry_run: bool) -> tuple[int, int, int, int]:
    """
    Filter rows from a single CSV file, removing any (asset, window_ts) groups
    with fewer than min_points rows.

    Returns (windows_removed, rows_removed).
    """
    df = storage.read_csv(path)
    if df.empty:
        return 0, 0, 0, 0

    grouped = df.groupby(["asset", "window_ts"]).size()
    small_windows = set(grouped[grouped < min_points].index.tolist())
    total_windows = int(len(grouped))
    kept_windows = total_windows - len(small_windows)

    keys = list(zip(df["asset"], df["window_ts"]))
    keep_mask = [k not in small_windows for k in keys]
    kept_df = df[keep_mask]
    rows_removed = int(len(df) - len(kept_df))
    windows_removed = len(small_windows)

    if not dry_run and small_windows:
        if storage.is_s3_uri(path):
            raise ValueError("Writing filtered CSVs back to S3 is not supported by this script yet")
        kept_df.to_csv(path, index=False)

    return windows_removed, rows_removed, kept_windows, int(len(kept_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter incomplete trading windows from price CSVs.")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS_DEFAULT,
                        help=f"Minimum data points per window (default: {MIN_POINTS_DEFAULT})")
    parser.add_argument("--prices-dir", default=storage.default_data_location("prices", "data/prices"),
                        help="Directory containing price CSV files (default: data/prices)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without modifying files")
    args = parser.parse_args()

    csv_files = storage.list_csv_paths(args.prices_dir, "prices_*.csv")
    if not csv_files:
        print(f"No price CSV files found in {args.prices_dir}")
        return

    total_windows_removed = 0
    total_rows_removed = 0
    total_windows_kept = 0
    total_rows_kept = 0
    for csv_file in csv_files:
        windows_removed, rows_removed, windows_kept, rows_kept = filter_csv(str(csv_file), args.min_points, args.dry_run)
        verb = "remove" if args.dry_run else "removed"
        name = str(csv_file).rsplit("/", 1)[-1]
        print(
            f"{name}: "
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
