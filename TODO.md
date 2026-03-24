# TODO

## Overfitting / Validation

- [ ] **Time-based train/test split** — sort sessions chronologically, run grid search on first 70%, evaluate the winning params on the last 30%. If edge holds out-of-sample it's a real signal. Recency matters — markets have regime changes so the test set should be the most recent data.

- [ ] **Bootstrap stability** — resample sessions with replacement N times (e.g. 100), re-run the grid search each time, record the winning (buy, sell, fill_window) region. If the winner consistently lands near the same spot it's a robust signal; if it jumps around wildly you're fitting noise.
