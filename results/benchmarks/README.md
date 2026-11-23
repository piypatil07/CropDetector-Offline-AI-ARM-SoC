# CPU model benchmarking CSVs

Source files were supplied from earlier on-device development runs, and are copied without recomputing. `legacy_100_run_summary.csv` used 100 measurements per configuration in its original collection; `optimized_40_run_summary.csv` has 40 measurements per configuration and additional preprocessing/output breakdowns. Individual thermal conditions and CPU thread settings are represented in the rows.

**Do not directly compare different versions of the benchmark harness or combine thermal status groups to claim a hardware speedup.** The engineer can run an in-app benchmark by long-pressing Scan to open the engineering menu; detailed timing/thermal behavior is in `CropBenchmark.kt`.
