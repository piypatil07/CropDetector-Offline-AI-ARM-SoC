# Device observations — 23 November 2026

These CSVs were parsed from the owner's filtered Android logs in a separate validation session. Confirmed device settings at test setup: **Airplane mode OFF, Wi-Fi ON**, Samsung SM-A146B / Android 15 / arm64-v8a. A later thermal snapshot recorded Android **thermal status 3 (severe)**. These timings are not a controlled heat-free experiment and do **not** document disconnected operation.

- `camera_scans_2026-11-23.csv`: Nine scans, one rejected Stage A input, eight Stage C results. No labeled ground truth was recorded.
- `llama_runs_2026-11-23.csv`: Two completed advisory generations with app-level elapsed times; one given a withheld condition, the other an accepted possible maize condition.

For completeness, request-date users previously **reported** a separate disconnected generation with Wi-Fi OFF and Airplane ON. Its metrics are documented in `docs/RESULTS_AND_LIMITATIONS.md` and should not be mixed with these logs.
