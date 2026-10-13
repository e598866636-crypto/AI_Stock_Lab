# TQAI Pro Enterprise v2.9.12 — Changelog

## Data-quality and research integrity

- Added `engines/data_quality_engine.py` to inspect local DB freshness, per-ticker staleness, OHLCV missing/invalid values, intraday estimates, sync metadata, stock-directory availability, historical shareholding snapshot availability, event CSV uniqueness, unknown ETF classification, and consistency between the event CSV and saved summary. It makes no network requests and does not mutate source data.
- Added `python -m research.data_quality_report` to produce JSON and Markdown health reports. `--strict` returns a non-zero exit code when the report is not `ok`.
- The Streamlit application now shows a data-quality banner at startup when locally stored data is stale or incomplete.
- Rebuilt the packaged breakout-event summary from the actual CSV. The current event dataset contains 186 unique events over 2024-12-06 to 2026-07-23 across 24 tickers, which is `CAUTION` under the existing workflow gate, not evidence of predictive power.
- Summary statistics now parse nullable booleans explicitly; unknown ETF status is shown as unknown and is not counted as confirmed non-ETF. Unique `(ticker, event_date)` keys are used for summary metrics while duplicate row counts remain visible.
- Custom event CSVs no longer overwrite the packaged production summary unless an explicit summary output path is provided.
- Per-ticker event rewrites are atomic and idempotent, avoiding duplicate appends on rerun. A successful re-scan replaces a ticker's old records; an empty response or exception is recorded as a failure and not marked processed.
- Full-universe inventory no longer silently falls back to the default watchlist when the local stock directory is empty; it fails with an actionable message.
- Checkpoint writes use atomic file replacement and track `failed_tickers` for retry on a later resume.

## Decision audit and safety gate

- Directional dimensions marked `資料不足` no longer count as neutral votes or inflate the denominator of consensus agreement.
- Added `potential_dims`, `data_coverage_pct`, and `unavailable_dims` to decision results and surfaced coverage in the Streamlit consensus panel.
- BUY READY / AVOID conclusions require at least 60% of directional dimensions to have valid data. Missing risk or liquidity checks force WATCH; a confirmed hard veto still takes precedence.
- Decision descriptions explicitly state that fixed rules are not validated probabilities.

## Tests and limits

- Added tests for data-health statuses, report consistency, duplicate event handling, nullable ETF classification, full-universe fallback prevention, failed-ticker retry behavior, and decision data-coverage gates.
- Final offline test suite: 74 passed.
- `compileall` passed.
- Golden regression could not run because `tests/golden/raw/` contains no frozen `.parquet` inputs. Real online data sync and out-of-sample predictive performance remain unverified.
