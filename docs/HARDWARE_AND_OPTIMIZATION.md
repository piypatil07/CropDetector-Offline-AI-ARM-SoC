# ARM hardware utilization and optimization

## Tested platform

- Samsung Galaxy A14 5G **SM-A146B**, Exynos 1330 ARM-based mobile SoC, `arm64-v8a`, Android 15.
- Physical memory from earlier device inspection: approximately **3.37 GiB usable RAM**. This is not identical to free memory or process RSS.
- The 23 Nov preflight snapshot showed ~**445 MB MemAvailable** and only **1.4 GB free flash storage**; device state varies.

## CPU inference

- Plant validator: MobileNetV2 width 0.35, 160-pixel input, integer quantization and a cheap rejection gate.
- Disease classifier: 224-pixel MobileNetV2; bundled FP32 / FP16 / INT8 formats allow controlled precision/timing comparisons. The main UI defaults to INT8 with 2 CPU threads; long-press Scan reveals the engineering benchmark menu.
- LLM: ExecuTorch Android `1.5.1`, Llama 3.2 1B Instruct SpinQuant INT4 pre-exported model; CPU generated text with model reload per request. Code requests `maxNewTokens(96)`, but measured tokenizer stats may report a different count; report observed tokens, not just configured values.
- CameraX, Kotlin and Android XML handle capture and presentation; no Android `INTERNET` permission is declared.

## Not measured / not claimed

- No verified NPU/GPU acceleration or per-core cycle counters are reported. This is principally an **ARM CPU deployment**.
- Process RSS in ExecuTorch logs (e.g. ~878.5 MiB at end of generation on 23 Nov) is not the same metric or time point as a later `dumpsys meminfo` PSS (~140 MiB after inference). Neither alone establishes peak combined memory usage.
- Thermal status **3** in the later snapshot indicates **severe throttling**; repeat decoding speeds cannot be compared as thermally controlled trials.
- The LLM's logged "Model Load Time: 0" within generation stats does not imply zero real startup overhead. Use the app elapsed time and timestamps for complete advisory time.

See `results/benchmarks/` for supplied earlier engineering CSVs. Some runs have different benchmark versions and thermal profiles; do not combine them as an apples-to-apples speedup without filtering protocols.
